#!/usr/bin/env python3
"""Build article-disjoint Wikipedia IME task groups for the D1 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beme.converter import Converter  # noqa: E402
from beme.ipadic import katakana_to_hiragana  # noqa: E402

DEFAULT_DATASET = "wikimedia/wikipedia"
DEFAULT_CONFIG = "20231101.ja"
DEFAULT_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
DEFAULT_LICENSE = "CC BY-SA 3.0 and GFDL"
DEFAULT_DICTIONARY = REPOSITORY_ROOT / "data" / "dictionary.tsv"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "d1"
TARGETS = {"train": 10_000, "dev": 1_000, "test": 2_000}
_SPLIT_RANGES = (("train", 7700), ("dev", 8500), ("test", 10_000))
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？])|[\r\n]+")
_KANA = re.compile(r"^[ぁ-ゖー]+$")
_SPACE_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Article:
    article_id: str
    title: str
    text: str
    url: str = ""


def normalize_text(text: str) -> str:
    return _SPACE_PUNCT.sub("", unicodedata.normalize("NFKC", text)).casefold()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def article_split(article_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{article_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    for split, upper in _SPLIT_RANGES:
        if bucket < upper:
            return split
    raise AssertionError("unreachable split bucket")


def _feature_fields(node: object) -> Sequence[str]:
    feature = node.feature
    return tuple(feature.split(",") if isinstance(feature, str) else feature)


def node_reading(node: object) -> str | None:
    fields = _feature_fields(node)
    if len(fields) < 8 or fields[7] == "*":
        return None
    reading = katakana_to_hiragana(fields[7])
    return reading if _KANA.fullmatch(reading) else None


def load_exclusion_terms(repo_root: Path = REPOSITORY_ROOT) -> tuple[str, ...]:
    """Load evaluation text fingerprints used to reject Wikipedia overlaps.

    Only context-bearing strings of at least 12 normalized characters are used;
    isolated candidate surfaces are deliberately not excluded because shared
    vocabulary is not document leakage.
    """

    terms: set[str] = set()
    ajimee_path = repo_root / "data" / "ajimee_bench_jwtd_v2_v1.json"
    if ajimee_path.exists():
        for case in json.loads(ajimee_path.read_text(encoding="utf-8")):
            for key in ("original_text", "context_text"):
                value = case.get(key)
                if isinstance(value, str):
                    normalized = normalize_text(value)
                    if len(normalized) >= 12:
                        terms.add(normalized)
    handmade_path = repo_root / "data" / "evaluation_cases.jsonl"
    if handmade_path.exists():
        for line in handmade_path.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            joined = f"{case.get('preceding_context', '')}{case.get('following_context', '')}"
            normalized = normalize_text(joined)
            if len(normalized) >= 12:
                terms.add(normalized)
    return tuple(sorted(terms, key=lambda term: (-len(term), term)))


def overlaps_evaluation(text: str, exclusion_terms: Sequence[str]) -> bool:
    normalized = normalize_text(text)
    return any(term in normalized for term in exclusion_terms)


def iter_hf_articles(dataset: str, config: str, revision: str) -> Iterator[Article]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("datasets is required; run 'uv sync --group dev'") from error
    rows = load_dataset(dataset, config, split="train", streaming=True, revision=revision)
    for row in rows:
        article_id, text = row.get("id"), row.get("text")
        if isinstance(article_id, str) and isinstance(text, str) and text:
            yield Article(article_id, str(row.get("title", "")), text, str(row.get("url", "")))


def iter_jsonl_articles(path: Path) -> Iterator[Article]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            try:
                yield Article(str(row["id"]), str(row.get("title", "")), str(row["text"]), str(row.get("url", "")))
            except (KeyError, TypeError) as error:
                raise ValueError(f"invalid article on line {line_number}") from error


def sentence_nodes(tagger: object, sentence: str) -> list[tuple[object, int, int]]:
    positioned: list[tuple[object, int, int]] = []
    cursor = 0
    for node in tagger(sentence):
        surface = node.surface
        start = sentence.find(surface, cursor)
        if start < 0:
            continue
        end = start + len(surface)
        positioned.append((node, start, end))
        cursor = end
    return positioned

def article_surface_frequencies(tagger: object, text: str) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for node in tagger(text):
        if node_reading(node):
            frequencies[node.surface] += 1
    return frequencies


def article_groups(
    article: Article,
    split: str,
    converter: Converter,
    tagger: object,
    context_chars: int,
    max_span_tokens: int,
    candidate_k: int,
    multi_span_every: int,
    max_reading_chars: int,
    ambiguous_surfaces: set[str],
    candidate_cache: dict[str, tuple[list[str], list[float]] | None],
) -> tuple[list[dict[str, object]], Counter[str]]:
    groups: list[dict[str, object]] = []
    surface_frequency: Counter[str] = Counter()
    article_hash = sha256_text(article.text)
    for sentence_index, sentence in enumerate(_SENTENCE_BOUNDARY.split(article.text)):
        sentence = sentence.strip()
        if not sentence:
            continue
        nodes = sentence_nodes(tagger, sentence)
        for node, _, _ in nodes:
            if node_reading(node):
                surface_frequency[node.surface] += 1
        for start_index in range(len(nodes)):
            readings: list[str] = []
            for span_size in range(1, max_span_tokens + 1):
                if span_size > 1:
                    sample_key = hashlib.sha256(
                        f"{article.article_id}:{sentence_index}:{start_index}".encode()
                    ).digest()
                    if int.from_bytes(sample_key[:4], "big") % multi_span_every:
                        break
                end_index = start_index + span_size
                if end_index > len(nodes):
                    break
                node, start, _ = nodes[start_index]
                last_node, _, end = nodes[end_index - 1]
                if last_node.surface in ambiguous_surfaces:
                    break
                if len(last_node.surface) == 1 and "\u4e00" <= last_node.surface <= "\u9fff":
                    break
                reading = node_reading(last_node)
                if reading is None:
                    break
                if not any(
                    entry.surface == last_node.surface
                    for entry in converter.dictionary.entries_for_reading(reading)
                ):
                    break
                readings.append(reading)
                combined_reading = "".join(readings)
                if len(combined_reading) > max_reading_chars:
                    break
                positive = sentence[start:end]
                if positive == combined_reading or len(combined_reading) < 2:
                    continue
                if combined_reading not in candidate_cache:
                    try:
                        result = converter.convert(combined_reading, k=candidate_k)
                    except ValueError:
                        candidate_cache[combined_reading] = None
                    else:
                        surfaces = [candidate.surface for candidate in result.candidates]
                        costs = [candidate.cost for candidate in result.candidates]
                        candidate_cache[combined_reading] = (surfaces, costs) if len(surfaces) >= 2 else None
                cached = candidate_cache[combined_reading]
                if cached is None:
                    continue
                candidates, costs = cached
                if positive not in candidates:
                    continue
                group_id = hashlib.sha256(
                    f"{article.article_id}\0{sentence_index}\0{start}\0{end}\0{combined_reading}\0{positive}".encode()
                ).hexdigest()[:20]
                groups.append(
                    {
                        "id": group_id,
                        "source": DEFAULT_CONFIG,
                        "article_id": article.article_id,
                        "article_sha256": article_hash,
                        "split": split,
                        "left_context": sentence[max(0, start - context_chars) : start],
                        "reading": combined_reading,
                        "right_context": sentence[end : end + context_chars],
                        "positive": positive,
                        "candidates": candidates,
                        "viterbi_costs": costs,
                        "gold_index": candidates.index(positive),
                        "span_tokens": span_size,
                        "context_truncated": start > context_chars or len(sentence) - end > context_chars,
                        "contrast_pair_id": None,
                    }
                )
    return groups, surface_frequency

def group_fingerprint(group: Mapping[str, object]) -> str:
    text = unicodedata.normalize(
        "NFKC",
        f"{group.get('left_context', '')}\0{group['reading']}\0{group['positive']}\0{group.get('right_context', '')}",
    )
    normalized = "".join(character for character in text.casefold() if not character.isspace())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def balanced_select(
    groups: Iterable[dict[str, object]],
    target: int,
    per_reading: int,
    seed: int,
    forbidden_fingerprints: set[str] | None = None,
) -> list[dict[str, object]]:
    by_reading: dict[str, list[dict[str, object]]] = defaultdict(list)
    forbidden = forbidden_fingerprints or set()
    for group in groups:
        if group_fingerprint(group) not in forbidden:
            by_reading[str(group["reading"])].append(group)
    rng = random.Random(seed)
    for values in by_reading.values():
        values.sort(key=lambda group: str(group["id"]))
        rng.shuffle(values)
    readings = sorted(by_reading)
    rng.shuffle(readings)

    # Reserve 10% of items for two-context contrast pairs before the
    # reading-balanced round robin consumes one item from every reading.
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    reading_counts: Counter[str] = Counter()
    pair_target = target // 20
    pair_count = 0
    for reading in readings:
        by_positive: dict[str, list[dict[str, object]]] = defaultdict(list)
        for group in by_reading[reading]:
            by_positive[str(group["positive"])].append(group)
        positives = sorted(by_positive)
        if len(positives) < 2:
            continue
        for positive in positives[:2]:
            group = min(by_positive[positive], key=lambda item: str(item["id"]))
            selected.append(group)
            selected_ids.add(str(group["id"]))
            reading_counts[reading] += 1
        pair_count += 1
        if pair_count == pair_target:
            break

    for round_index in range(per_reading):
        for reading in readings:
            if reading_counts[reading] >= per_reading:
                continue
            values = by_reading[reading]
            if round_index >= len(values):
                continue
            group = values[round_index]
            if str(group["id"]) in selected_ids:
                continue
            selected.append(group)
            selected_ids.add(str(group["id"]))
            reading_counts[reading] += 1
            if len(selected) == target:
                return sorted(selected, key=lambda item: str(item["id"]))
    return sorted(selected, key=lambda item: str(item["id"]))


def add_contrast_pairs(groups: list[dict[str, object]]) -> int:
    by_reading: dict[str, list[dict[str, object]]] = defaultdict(list)
    for group in groups:
        by_reading[str(group["reading"])].append(group)
    pair_count = 0
    for reading, values in sorted(by_reading.items()):
        by_positive: dict[str, list[dict[str, object]]] = defaultdict(list)
        for group in values:
            by_positive[str(group["positive"])].append(group)
        positives = sorted(by_positive)
        if len(positives) < 2:
            continue
        left = min(by_positive[positives[0]], key=lambda group: str(group["id"]))
        right = min(by_positive[positives[1]], key=lambda group: str(group["id"]))
        pair_id = hashlib.sha256(f"{reading}\0{left['id']}\0{right['id']}".encode()).hexdigest()[:20]
        left["contrast_pair_id"] = pair_id
        right["contrast_pair_id"] = pair_id
        pair_count += 1
    return pair_count


def frequency_bucket(count: int, ordered_counts: Sequence[int]) -> str:
    if not ordered_counts:
        return "unknown"
    low = ordered_counts[len(ordered_counts) // 3]
    high = ordered_counts[(2 * len(ordered_counts)) // 3]
    if count <= low:
        return "low"
    if count <= high:
        return "mid"
    return "high"


def add_frequency_buckets(groups: Iterable[dict[str, object]], frequencies: Mapping[str, int]) -> None:
    counts = sorted(frequencies.values())
    for group in groups:
        positive = str(group["positive"])
        candidates = list(group["candidates"])
        costs = list(group["viterbi_costs"])
        hard_negative = min(
            ((candidate, cost) for candidate, cost in zip(candidates, costs) if candidate != positive),
            key=lambda item: (item[1], item[0]),
        )[0]
        group["frequency_bucket"] = {
            "positive": frequency_bucket(frequencies.get(positive, 0), counts),
            "hard_negative": frequency_bucket(frequencies.get(hard_negative, 0), counts),
        }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict[str, object]:
    try:
        from fugashi import GenericTagger
        import ipadic
    except ImportError as error:
        raise RuntimeError("fugashi and ipadic are required; run 'uv sync --group dev'") from error

    converter = Converter.from_tsv(args.dictionary, beam_width=args.beam_width)
    surface_readings: dict[str, set[str]] = defaultdict(set)
    for entry in converter.dictionary.entries:
        surface_readings[entry.surface].add(entry.reading)
    ambiguous_surfaces = {
        surface for surface, readings in surface_readings.items() if len(readings) > 1
    }
    tagger = GenericTagger(f"-r /dev/null -d {ipadic.DICDIR}")
    exclusion_terms = load_exclusion_terms()
    source = iter_jsonl_articles(args.articles_jsonl) if args.articles_jsonl else iter_hf_articles(args.dataset, args.config, args.revision)
    pools: dict[str, list[dict[str, object]]] = defaultdict(list)
    article_records: dict[str, dict[str, object]] = {}
    frequencies: Counter[str] = Counter()
    candidate_cache: dict[str, tuple[list[str], list[float]] | None] = {}
    bytes_seen = 0
    excluded_articles = 0
    processed_articles = 0
    sampled_articles = 0

    pool_targets = {split: int(target * args.pool_multiplier) for split, target in TARGETS.items()}
    for article in source:
        encoded_bytes = len(article.text.encode("utf-8"))
        if bytes_seen and bytes_seen + encoded_bytes > args.max_source_bytes:
            break
        bytes_seen += encoded_bytes
        processed_articles += 1
        if overlaps_evaluation(article.text, exclusion_terms):
            excluded_articles += 1
            continue
        split = article_split(article.article_id, args.seed)
        sample_key = hashlib.sha256(f"{args.seed}:{article.article_id}".encode()).digest()
        pools_ready = all(len(pools[name]) >= pool_targets[name] for name in TARGETS)
        sample_for_groups = (
            not pools_ready
            or int.from_bytes(sample_key[:4], "big") % args.article_sample_every == 0
        )
        if sample_for_groups:
            sampled_articles += 1
            groups, article_frequencies = article_groups(
                article,
                split,
                converter,
                tagger,
                args.context_chars,
                args.max_span_tokens,
                args.candidate_k,
                args.multi_span_every,
                args.max_reading_chars,
                ambiguous_surfaces,
                candidate_cache,
            )
        else:
            groups = []
            article_frequencies = article_surface_frequencies(tagger, article.text)
        frequencies.update(article_frequencies)
        if groups:
            pools[split].extend(groups)
            limit = pool_targets[split]
            if len(pools[split]) > int(limit * 1.1):
                pools[split].sort(
                    key=lambda group: hashlib.sha256(
                        f"{args.seed}:{group['id']}".encode()
                    ).digest()
                )
                del pools[split][limit:]
            article_records[article.article_id] = {
                "article_id": article.article_id,
                "title": article.title,
                "url": article.url,
                "sha256": sha256_text(article.text),
                "utf8_bytes": encoded_bytes,
                "split": split,
            }
        if processed_articles % 100 == 0:
            counts = " ".join(f"{name}={len(pools[name])}" for name in TARGETS)
            print(
                f"progress articles={processed_articles} sampled={sampled_articles} bytes={bytes_seen} {counts}",
                file=sys.stderr,
                flush=True,
            )

    selected: dict[str, list[dict[str, object]]] = {}
    shortages: dict[str, tuple[int, int]] = {}
    locked_fingerprints: set[str] = set()
    split_seed_offsets = {"train": 0, "dev": 1, "test": 2}
    for split in ("test", "dev", "train"):
        target = TARGETS[split]
        selected[split] = balanced_select(
            pools[split],
            target,
            args.per_reading,
            args.seed + split_seed_offsets[split],
            locked_fingerprints,
        )
        locked_fingerprints.update(group_fingerprint(group) for group in selected[split])
        if len(selected[split]) < target:
            shortages[split] = (len(selected[split]), target)
    if shortages:
        details = ", ".join(f"{split}={actual}/{target}" for split, (actual, target) in shortages.items())
        raise RuntimeError(f"insufficient balanced groups: {details}; increase --max-source-bytes or --per-reading")

    selected_article_ids = {str(group["article_id"]) for rows in selected.values() for group in rows}
    pair_counts = {split: add_contrast_pairs(rows) for split, rows in selected.items()}
    for rows in selected.values():
        add_frequency_buckets(rows, frequencies)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_hashes = {
        split: write_jsonl(args.output_dir / f"{split}.jsonl", rows) for split, rows in selected.items()
    }
    audit_rng = random.Random(args.seed)
    audit_pool = selected["train"].copy()
    audit_rng.shuffle(audit_pool)
    audit_rows = [
        {**group, "audit_label_correct": None, "audit_note": ""}
        for group in audit_pool[: args.audit_size]
    ]
    artifact_hashes["audit"] = write_jsonl(args.output_dir / "audit.jsonl", audit_rows)
    article_manifest = sorted(
        (article_records[article_id] for article_id in selected_article_ids),
        key=lambda row: (str(row["split"]), str(row["article_id"])),
    )
    artifact_hashes["articles"] = write_jsonl(args.output_dir / "articles.jsonl", article_manifest)
    dictionary_sha = hashlib.sha256(args.dictionary.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "created_on": "2026-08-22",
        "source": {
            "dataset": args.dataset,
            "config": args.config,
            "revision": args.revision,
            "license": DEFAULT_LICENSE,
            "license_url": "https://huggingface.co/datasets/wikimedia/wikipedia",
            "source_utf8_bytes_scanned": bytes_seen,
            "articles_processed": processed_articles,
            "articles_excluded_for_evaluation_overlap": excluded_articles,
            "articles_sampled_for_task_groups": sampled_articles,
        },
        "split": {"algorithm": "sha256(seed:article_id) modulo 10000", "seed": args.seed, "ranges": _SPLIT_RANGES},
        "counts": {split: len(rows) for split, rows in selected.items()},
        "pool_counts": {split: len(pools[split]) for split in TARGETS},
        "unique_articles": {split: len({row["article_id"] for row in rows}) for split, rows in selected.items()},
        "unique_readings": {split: len({row["reading"] for row in rows}) for split, rows in selected.items()},
        "contrast_pairs": pair_counts,
        "evaluation_overlap": {
            "method": "normalized AJIMEE original/context and handmade context substring exclusion",
            "term_count": len(exclusion_terms),
            "minimum_normalized_characters": 12,
            "mozc_note": "Mozc provides readings and assertions, not source documents; shared vocabulary is not treated as document leakage.",
        },
        "candidate_generation": {
            "cross_split_normalized_group_deduplication": True,
            "deduplication_precedence": ["test", "dev", "train"],
            "dictionary": str(args.dictionary),
            "dictionary_sha256": dictionary_sha,
            "beam_width": args.beam_width,
            "candidate_k": args.candidate_k,
            "max_span_tokens": args.max_span_tokens,
            "multi_span_sample_every": args.multi_span_every,
            "article_sample_every_after_pool_fill": args.article_sample_every,
            "max_reading_characters": args.max_reading_chars,
            "ambiguous_surface_readings_excluded": True,
            "ambiguous_surface_count": len(ambiguous_surfaces),
            "single_cjk_surface_excluded": True,
            "per_reading_cap": args.per_reading,
            "context_characters_each_side": args.context_chars,
        },
        "artifacts_sha256": artifact_hashes,
        "audit": {"sample_size": len(audit_rows), "seed": args.seed, "completed": False},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--articles-jsonl", type=Path, help="offline source with id/title/text/url fields")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-source-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--context-chars", type=int, default=48)
    parser.add_argument("--max-span-tokens", type=int, default=3)
    parser.add_argument("--multi-span-every", type=int, default=20)
    parser.add_argument("--article-sample-every", type=int, default=5)
    parser.add_argument("--max-reading-chars", type=int, default=16)
    parser.add_argument("--per-reading", type=int, default=8)
    parser.add_argument("--pool-multiplier", type=float, default=2.0)
    parser.add_argument("--audit-size", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if any(
        value <= 0
        for value in (
            args.max_source_bytes,
            args.per_reading,
            args.audit_size,
            args.multi_span_every,
            args.max_reading_chars,
            args.article_sample_every,
        )
    ):
        raise SystemExit("source bytes, caps, sample interval, and audit size must be positive")
    if args.candidate_k > args.beam_width:
        raise SystemExit("--candidate-k cannot exceed --beam-width")
    manifest = build(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True) if args.json else f"Wrote D1 data to {args.output_dir}")
    return 0


if __name__ == "__main__":
    status = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # datasets streaming can leave an Arrow downloader thread alive at interpreter shutdown.
    os._exit(status)
