#!/usr/bin/env python3
"""Build the (reading, surface) corpus frequency table from train-split articles only.

Leakage policy (PLAN §3.4, issue #19 comment §4):
* Split assignment is imported from build_d1_data.article_split -- never re-derived.
* Articles overlapping the evaluation suites (AJIMEE / handmade) are excluded via
  build_d1_data.overlaps_evaluation before counting.
* After counting, every counted article id is checked against the article ids
  actually present in data/d1/dev.jsonl and data/d1/test.jsonl; any intersection
  aborts the build with a non-zero exit.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
SRC_ROOT = REPOSITORY_ROOT / "src"
for path in (str(SCRIPTS_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from build_d1_data import (  # noqa: E402
    article_split,
    iter_hf_articles,
    iter_jsonl_articles,
    load_exclusion_terms,
    node_reading,
    overlaps_evaluation,
)
from beme.corpus_freq import CorpusFrequencyTable  # noqa: E402

DEFAULT_DATASET = "wikimedia/wikipedia"
DEFAULT_CONFIG = "20231101.ja"
DEFAULT_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
DEFAULT_SPLIT_SEED = 20260822
DEFAULT_TARGET_BYTES = 100 * 1024 * 1024
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "corpus_freq_v1.bin"
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data" / "corpus_freq_v1.manifest.json"
DEFAULT_D1_DIR = REPOSITORY_ROOT / "data" / "d1"
DEFAULT_COVERAGE_PAIRS = (
    REPOSITORY_ROOT / "artifacts" / "issue19" / "s-measurement" / "coverage_pairs.jsonl"
)


def _make_tagger():
    try:
        from fugashi import GenericTagger
        import ipadic
    except ImportError as error:
        raise RuntimeError("fugashi and ipadic are required; run 'uv sync --group dev'") from error
    return GenericTagger(f"-r /dev/null -d {ipadic.DICDIR}")


def _d1_dev_test_ids(d1_dir: Path) -> set[str]:
    ids: set[str] = set()
    for name in ("dev", "test"):
        path = d1_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    ids.add(json.loads(line)["article_id"])
    return ids


def _coverage_from_pairs(
    pairs_path: Path, table: CorpusFrequencyTable, d1_dir: Path
) -> dict[str, object]:
    """Fill the coverage block from task1's coverage_pairs.jsonl (no d1 rescan)."""
    if not pairs_path.exists():
        return {
            "error": f"coverage_pairs.jsonl not found: {pairs_path}",
            "d1_test_gold_pairs": 0,
            "d1_test_gold_pairs_seen": 0,
            "d1_test_gold_pair_coverage": 0.0,
            "d1_test_gold_surfaces_seen": 0,
            "d1_test_gold_surface_coverage": 0.0,
            "d1_train_gold_pair_coverage": 0.0,
        }
    test_pairs: Counter[tuple[str, str]] = Counter()
    train_pairs: Counter[tuple[str, str]] = Counter()
    for split, target in (("test", test_pairs), ("train", train_pairs)):
        path = d1_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                target[(row["reading"], row["positive"])] += 1
    test_total = sum(test_pairs.values())
    train_total = sum(train_pairs.values())
    test_seen_pairs = sum(
        count for pair, count in test_pairs.items() if table.pair_count(*pair) > 0
    )
    test_seen_surfaces = sum(
        count for pair, count in test_pairs.items() if table.pair_count(*pair) > 0
    )
    train_seen_pairs = sum(
        count for pair, count in train_pairs.items() if table.pair_count(*pair) > 0
    )
    return {
        "d1_test_gold_pairs": test_total,
        "d1_test_gold_pairs_seen": test_seen_pairs,
        "d1_test_gold_pair_coverage": test_seen_pairs / test_total if test_total else 0.0,
        "d1_test_gold_surfaces_seen": test_seen_surfaces,
        "d1_test_gold_surface_coverage": (
            test_seen_surfaces / test_total if test_total else 0.0
        ),
        "d1_train_gold_pair_coverage": train_seen_pairs / train_total if train_total else 0.0,
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    tagger = _make_tagger()
    exclusion_terms = load_exclusion_terms()
    source = (
        iter_jsonl_articles(args.articles_jsonl)
        if args.articles_jsonl
        else iter_hf_articles(args.dataset, args.config, args.revision)
    )

    pair_counts: Counter[tuple[str, str]] = Counter()
    surface_counts: Counter[str] = Counter()
    counted_article_ids: set[str] = set()
    total_tokens = 0
    bytes_seen = 0
    seen = 0
    train_counted = 0
    dev_test_skipped = 0
    evaluation_overlap_skipped = 0

    for article in source:
        if args.max_articles is not None and seen >= args.max_articles:
            break
        encoded_bytes = len(article.text.encode("utf-8"))
        if bytes_seen and bytes_seen + encoded_bytes > args.target_bytes:
            break
        # Skipped articles still count toward the scanned window (fetch_corpus.py
        # convention), so the window stays aligned with d1's.
        bytes_seen += encoded_bytes
        seen += 1
        if overlaps_evaluation(article.text, exclusion_terms):
            evaluation_overlap_skipped += 1
            continue
        if article_split(article.article_id, args.split_seed) != "train":
            dev_test_skipped += 1
            continue
        for node in tagger(article.text):
            reading = node_reading(node)
            if reading is None:
                continue
            pair_counts[(reading, node.surface)] += 1
            surface_counts[node.surface] += 1
            total_tokens += 1
        counted_article_ids.add(article.article_id)
        train_counted += 1
        if train_counted % 100 == 0:
            print(
                f"progress counted={train_counted} bytes={bytes_seen} "
                f"pairs={len(pair_counts)}",
                file=sys.stderr,
                flush=True,
            )

    table = CorpusFrequencyTable.from_counts(pair_counts, surface_counts, total_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table_sha256 = table.dump(args.output)

    # -- leak audit against the d1 dev/test article ids --------------------
    d1_ids = _d1_dev_test_ids(args.d1_dir)
    intersection = counted_article_ids & d1_ids
    leak_passed = not intersection
    manifest = {
        "schema_version": 1,
        "created_on": date.today().isoformat(),
        "source": {
            "dataset": args.dataset,
            "config": args.config,
            "revision": args.revision,
            "utf8_bytes_scanned": bytes_seen,
            "bytes_count_includes_skipped_articles": True,
        },
        "split": {
            "algorithm": "sha256(seed:article_id) modulo 10000",
            "seed": args.split_seed,
            "train_upper": 7700,
        },
        "articles": {
            "seen": seen,
            "train_counted": train_counted,
            "dev_test_skipped": dev_test_skipped,
            "evaluation_overlap_skipped": evaluation_overlap_skipped,
        },
        "counts": {
            "total_tokens": total_tokens,
            "unique_pairs": len(pair_counts),
            "unique_surfaces": len(surface_counts),
            "unique_readings": len({reading for reading, _ in pair_counts}),
        },
        "leak_audit": {
            "d1_dev_test_article_ids": len(d1_ids),
            "counted_article_ids_intersecting_d1_dev_test": len(intersection),
            "passed": leak_passed,
        },
        "coverage": _coverage_from_pairs(args.coverage_pairs, table, args.d1_dir),
        "table_sha256": table_sha256,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--target-bytes", type=int, default=DEFAULT_TARGET_BYTES)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--articles-jsonl", type=Path, default=None)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument(
        "--coverage-pairs", type=Path, default=DEFAULT_COVERAGE_PAIRS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target_bytes <= 0:
        raise SystemExit("--target-bytes must be positive")
    manifest = build(args)
    if not manifest["leak_audit"]["passed"]:  # type: ignore[index]
        raise SystemExit(
            "leak audit failed: counted articles intersect d1 dev/test; "
            f"see {args.manifest}"
        )
    print(f"Wrote {args.output} (sha256 {manifest['table_sha256']})")
    print(f"manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    status = main()
    sys.stdout.flush()
    sys.stderr.flush()
    import os

    os._exit(status)
