#!/usr/bin/env python3
"""Build a train-article-only (reading, surface) frequency table."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from build_d1_data import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_DATASET,
    DEFAULT_REVISION,
    _SPLIT_RANGES,
    article_split,
    iter_hf_articles,
    iter_jsonl_articles,
    load_exclusion_terms,
    node_reading,
    overlaps_evaluation,
)
from beme.corpus_freq import CorpusFrequencyTable  # noqa: E402


def _jsonl_pairs(path: Path) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    if not path.is_file():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            surface_key = "positive" if "positive" in row else "surface"
            counts[(str(row["reading"]), str(row[surface_key]))] += int(row.get("count", 1))
    return counts


def _coverage(table: CorpusFrequencyTable, d1_dir: Path, coverage_path: Path) -> tuple[dict[str, object], bool]:
    split_pairs = {name: _jsonl_pairs(d1_dir / f"{name}.jsonl") for name in ("train", "dev", "test")}
    test_pairs = split_pairs["test"]
    train_pairs = split_pairs["train"]

    def weighted(pairs: Counter[tuple[str, str]], predicate) -> tuple[int, int, float]:
        total = sum(pairs.values())
        seen = sum(count for pair, count in pairs.items() if predicate(pair))
        return seen, total, seen / total if total else 0.0

    pair_seen, pair_total, pair_rate = weighted(test_pairs, lambda pair: table.pair_count(*pair) > 0)
    surface_seen, _, surface_rate = weighted(test_pairs, lambda pair: table.surface_count(pair[1]) > 0)
    train_seen, train_total, train_rate = weighted(train_pairs, lambda pair: table.pair_count(*pair) > 0)
    span1_pairs: Counter[tuple[str, str]] = Counter()
    test_path = d1_dir / "test.jsonl"
    if test_path.is_file():
        for line in test_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                if row.get("span_tokens") == 1:
                    span1_pairs[(str(row["reading"]), str(row["positive"]))] += 1
    span1_seen, span1_total, span1_rate = weighted(span1_pairs, lambda pair: table.pair_count(*pair) > 0)

    coverage_match: bool | None = None
    coverage_sha: str | None = None
    if coverage_path.is_file():
        raw = coverage_path.read_bytes()
        coverage_sha = hashlib.sha256(raw).hexdigest()
        reference = _jsonl_pairs(coverage_path)
        coverage_match = reference == train_pairs + test_pairs
        if not coverage_match:
            raise RuntimeError("coverage-pairs.jsonl does not match d1 train/test gold pairs")
    result = {
        "d1_test_gold_pairs": pair_total,
        "d1_test_gold_pairs_seen": pair_seen,
        "d1_test_gold_pair_coverage": pair_rate,
        "d1_test_gold_surfaces_seen": surface_seen,
        "d1_test_gold_surface_coverage": surface_rate,
        "d1_train_gold_pair_coverage": train_rate,
        "d1_test_gold_pair_coverage_span1_only": span1_rate,
        "d1_test_gold_pair_coverage_span1_seen": span1_seen,
        "d1_test_gold_pair_coverage_span1_total": span1_total,
        "d1_train_gold_pairs": train_total,
        "source": "recomputed from data/d1/{train,test}.jsonl (reading, positive); cross-checked against coverage_pairs.jsonl",
        "coverage_pairs_sha256": coverage_sha,
        "coverage_pairs_match": coverage_match,
    }
    return result, bool(split_pairs["train"] or split_pairs["test"])


def _d1_audit(d1_dir: Path, counted: set[str], seed: int, allow_missing: bool) -> tuple[dict[str, object], bool]:
    paths = [d1_dir / f"{name}.jsonl" for name in ("train", "dev", "test")]
    available = all(path.is_file() for path in paths)
    if available:
        d1_ids = {
            str(json.loads(line)["article_id"])
            for path in (paths[1], paths[2])
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        }
        intersection = counted & d1_ids
        misclassified = {article_id for article_id in d1_ids if article_split(article_id, seed) == "train"}
    else:
        d1_ids, intersection, misclassified = set(), set(), set()
    passed = available and not intersection and not misclassified
    audit = {
        "d1_dev_test_article_ids": len(d1_ids),
        "counted_article_ids_intersecting_d1_dev_test": len(intersection),
        "d1_dev_test_article_ids_misclassified_as_train": len(misclassified),
        "d1_available": available,
        "passed": passed,
    }
    return audit, passed or (allow_missing and not available)


def build(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    try:
        from fugashi import GenericTagger
        import ipadic
    except ImportError as error:
        raise RuntimeError("fugashi and ipadic are required; run 'uv sync --group dev'") from error
    tagger = GenericTagger(f"-r /dev/null -d {ipadic.DICDIR}")
    exclusion_terms = load_exclusion_terms()
    source = iter_jsonl_articles(args.articles_jsonl) if args.articles_jsonl else iter_hf_articles(args.dataset, args.config, args.revision)
    pair_counts: Counter[tuple[str, str]] = Counter()
    surface_counts: Counter[str] = Counter()
    counted_ids: set[str] = set()
    bytes_seen = articles_seen = train_counted = overlap_skipped = dev_test_skipped = total_tokens = 0
    started = time.perf_counter()
    for article in source:
        encoded_bytes = len(article.text.encode("utf-8"))
        if bytes_seen and bytes_seen + encoded_bytes > args.target_bytes:
            break
        bytes_seen += encoded_bytes
        articles_seen += 1
        if overlaps_evaluation(article.text, exclusion_terms):
            overlap_skipped += 1
            continue
        if article_split(article.article_id, args.split_seed) != "train":
            dev_test_skipped += 1
            continue
        train_counted += 1
        counted_ids.add(article.article_id)
        for node in tagger(article.text):
            reading = node_reading(node)
            if reading is None:
                continue
            pair_counts[(reading, node.surface)] += 1
            surface_counts[node.surface] += 1
            total_tokens += 1
        if args.progress_every and articles_seen % args.progress_every == 0:
            print(f"progress articles={articles_seen} train={train_counted} bytes={bytes_seen}", file=sys.stderr, flush=True)
        if args.max_articles is not None and articles_seen >= args.max_articles:
            break
    table = CorpusFrequencyTable.from_counts(pair_counts, surface_counts, total_tokens)
    table_sha = table.dump(args.output)
    audit, audit_status = _d1_audit(args.d1_dir, counted_ids, args.split_seed, args.allow_missing_d1)
    coverage, _ = _coverage(table, args.d1_dir, args.coverage_pairs)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    manifest = {
        "schema_version": 1,
        "created_on": str(date.today()),
        "source": {"dataset": args.dataset, "config": args.config, "revision": args.revision, "utf8_bytes_scanned": bytes_seen,
                   "byte_accounting": "build_d1_data: every article counted before filtering; break before exceeding target; articles are never truncated"},
        "split": {"algorithm": "sha256(seed:article_id) modulo 10000", "seed": args.split_seed, "train_upper": _SPLIT_RANGES[0][1]},
        "articles": {"seen": articles_seen, "train_counted": train_counted, "dev_test_skipped": dev_test_skipped, "evaluation_overlap_skipped": overlap_skipped},
        "counts": {"total_tokens": total_tokens, "unique_pairs": len(pair_counts), "unique_surfaces": len(surface_counts), "unique_readings": len(table._readings),
                   "marginals_consistent": sum(pair_counts.values()) == total_tokens and all(sum(count for (r, s), count in pair_counts.items() if s == surface) == count for surface, count in surface_counts.items())},
        "leak_audit": audit,
        "coverage": coverage,
        "table_sha256": table_sha,
        "environment": {"git_commit": git_commit, "python": platform.python_version(), "host": platform.node()},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest, 0 if audit_status else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "data/corpus_freq_v1.bin")
    parser.add_argument("--manifest", type=Path, default=REPOSITORY_ROOT / "data/corpus_freq_v1.manifest.json")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--target-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--max-articles", type=int)
    parser.add_argument("--split-seed", type=int, default=20260822)
    parser.add_argument("--articles-jsonl", type=Path)
    parser.add_argument("--d1-dir", type=Path, default=REPOSITORY_ROOT / "data/d1")
    parser.add_argument("--coverage-pairs", type=Path, default=REPOSITORY_ROOT / "artifacts/issue19/s-measurement/coverage_pairs.jsonl")
    parser.add_argument("--allow-missing-d1", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target_bytes <= 0 or args.progress_every < 0 or args.max_articles is not None and args.max_articles <= 0:
        raise SystemExit("target bytes and max articles must be positive; progress interval must be non-negative")
    manifest, status = build(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True) if args.json else f"Wrote corpus frequency table to {args.output}")
    return status


if __name__ == "__main__":
    args = sys.argv[1:]
    status = main(args)
    if "--articles-jsonl" not in args:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
    raise SystemExit(status)
