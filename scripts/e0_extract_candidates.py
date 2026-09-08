#!/usr/bin/env python3
"""Extract deterministic, sentence-level E0 candidates from D1 Wikipedia data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any, Iterable, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e0_leakage_common import (  # noqa: E402
    build_e0_leakage_index,
    check_e0_leakage,
    document_fingerprint,
    normalize_text,
)
from scripts.e0_quota_validator import quota_stratum  # noqa: E402

_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？])|[\r\n]+")
_HEADING_OR_LIST = re.compile(r"^(?:={1,6}.*={1,6}|#{1,6}\s|[*#;:|!]{1,2}\s|[-–—・●○▪︎]\s)")
_KANA = re.compile(r"^[ぁ-ゖー]+$")
_STRATA = (("1-8", 8), ("9-16", 16), ("17-32", 32), ("33+", None))


@dataclass(frozen=True)
class Article:
    article_id: str
    text: str
    title: str = ""
    url: str = ""


def split_sentences(text: str) -> list[str]:
    """Split exactly like D1, retaining the D1 sentence index space."""
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]


def _stratum(length: int) -> str:
    for label, upper in _STRATA:
        if upper is None or length <= upper:
            return label
    raise AssertionError("unreachable")


def _hash_key(seed: int, candidate: Mapping[str, Any]) -> bytes:
    normalized = str(candidate["normalized_surface"])
    value = f"{seed}{candidate['article_id']}{candidate['sentence_index']}{normalized}"
    return hashlib.sha256(value.encode("utf-8")).digest()


def select_candidates(
    candidates: Sequence[Mapping[str, Any]],
    target_count: int,
    seed: int,
    min_articles: int,
    max_per_article: int = 5,
    fill_quota: bool = False,
) -> list[dict[str, Any]]:
    """Select by per-stratum hash order, with article and per-article gates."""
    if target_count < 1 or max_per_article < 1:
        raise ValueError("target_count and max_per_article must be positive")
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        buckets[str(candidate["selection_stratum"])].append(candidate)
    for rows in buckets.values():
        rows.sort(key=lambda row: (_hash_key(seed, row), str(row["article_id"]), int(row["sentence_index"])))

    legacy_order = [label for label, _ in _STRATA if label in buckets]
    quota_order = ("2-4", "5-8", "9-16", "17+")
    order = [label for label in quota_order if label in buckets]
    order += [label for label in legacy_order if label not in order]
    selected: list[dict[str, Any]] = []
    seen_articles: set[str] = set()
    counts: defaultdict[str, int] = defaultdict(int)
    selected_keys: set[tuple[str, int, str]] = set()
    cursors = {label: 0 for label in order}
    quota_counts: defaultdict[str, int] = defaultdict(int)

    def take(row: Mapping[str, Any]) -> None:
        item = dict(row)
        item["selection_probability"] = 0.0
        selected.append(item)
        article_id = str(item["article_id"])
        selected_keys.add((article_id, int(item["sentence_index"]), str(item["normalized_surface"])))
        seen_articles.add(article_id)
        counts[article_id] += 1
        quota_counts[str(item["selection_stratum"])] += 1

    if fill_quota:
        desired = {"2-4": int(target_count * 0.10), "5-8": int(target_count * 0.25),
                   "9-16": int(target_count * 0.40)}
        desired["17+"] = target_count - sum(desired.values())
        for label in order:
            while len(selected) < target_count and quota_counts[label] < desired.get(label, 0):
                rows = buckets[label]
                while cursors[label] < len(rows):
                    row = rows[cursors[label]]
                    cursors[label] += 1
                    key = (str(row["article_id"]), int(row["sentence_index"]), str(row["normalized_surface"]))
                    if key in selected_keys or counts[str(row["article_id"])] >= max_per_article:
                        continue
                    take(row)
                    break
                else:
                    break

    # First obtain article coverage. The same hash order remains the tie-break.
    coverage = sorted(
        (row for label in order for row in buckets[label]),
        key=lambda row: (_hash_key(seed, row), str(row["article_id"])),
    )
    for row in coverage:
        if len(selected) >= target_count or len(seen_articles) >= min_articles:
            break
        if counts[str(row["article_id"])] == 0:
            take(row)

    while len(selected) < target_count:
        progressed = False
        for label in order:
            rows = buckets[label]
            while cursors[label] < len(rows):
                row = rows[cursors[label]]
                cursors[label] += 1
                key = (str(row["article_id"]), int(row["sentence_index"]), str(row["normalized_surface"]))
                if key in selected_keys:
                    continue
                if counts[str(row["article_id"])] < max_per_article:
                    take(row)
                    progressed = True
                    break
            if len(selected) >= target_count:
                break
        if not progressed:
            break

    if len(selected) < target_count:
        raise ValueError(f"insufficient candidates: {len(selected)} < {target_count}")
    if len(seen_articles) < min_articles:
        raise ValueError(f"insufficient articles: {len(seen_articles)} < {min_articles}")
    selected_by_stratum: defaultdict[str, int] = defaultdict(int)
    available_by_stratum = {label: len(rows) for label, rows in buckets.items()}
    for item in selected:
        selected_by_stratum[str(item["selection_stratum"])] += 1
    for item in selected:
        available = available_by_stratum[str(item["selection_stratum"])]
        item["selection_probability"] = max(
            1.0 / available,
            min(1.0, selected_by_stratum[str(item["selection_stratum"])] / available),
        )
    return selected


def _reading_from_node(node: Any) -> str:
    feature = getattr(node, "feature", None)
    if feature is None:
        return ""
    values = getattr(feature, "as_array", None)
    if callable(values):
        values = values()
    if values is None:
        values = feature.split(",") if isinstance(feature, str) else feature
    if values and len(values) > 7:
        reading = str(values[7])
        if reading and reading != "*":
            reading = "".join(chr(ord(char) - 0x60) if "ァ" <= char <= "ヺ" else char for char in reading)
            return reading if _KANA.fullmatch(reading) else ""
    return ""


def _literal(surface: str) -> bool:
    return all(unicodedata.category(char)[0] in {"P", "S"} for char in surface)


def _tokenize(sentence: str, tagger: Any) -> tuple[str, list[dict[str, Any]]] | None:
    tokens: list[dict[str, Any]] = []
    cursor = 0
    reading_cursor = 0
    for node in tagger(sentence):
        surface = str(node.surface)
        start = sentence.find(surface, cursor)
        if start < 0:
            return None
        end = start + len(surface)
        reading = _reading_from_node(node)
        is_literal = not reading and _literal(surface)
        if not reading and not is_literal:
            return None
        if is_literal:
            reading = surface
        token = {
            "surface": surface, "reading": reading,
            "surface_start": start, "surface_end": end,
            "reading_start": reading_cursor, "reading_end": reading_cursor + len(reading),
            "is_literal": is_literal,
        }
        tokens.append(token)
        cursor = end
        reading_cursor += len(reading)
    if not tokens or cursor != len(sentence):
        return None
    return "".join(token["reading"] for token in tokens), tokens


def _is_usable_sentence(sentence: str) -> bool:
    if not sentence or _HEADING_OR_LIST.match(sentence):
        return False
    if sentence.startswith(("{|", "|}", "<!--")):
        return False
    # A bare heading fragment generally has no sentence punctuation and no predicate.
    if len(sentence) < 2 or ("。" not in sentence and "！" not in sentence and "？" not in sentence and "\n" not in sentence):
        return False
    return True


def extract_article_candidates(article: Article, tagger: Any) -> list[dict[str, Any]]:
    article_hash = hashlib.sha256(article.text.encode("utf-8")).hexdigest()
    output: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    for sentence_index, sentence in enumerate(split_sentences(article.text)):
        if not _is_usable_sentence(sentence):
            continue
        tokenized = _tokenize(sentence, tagger)
        if tokenized is None:
            continue
        reading, tokens = tokenized
        if not reading:
            continue
        normalized_surface = normalize_text(sentence)
        if normalized_surface in seen_fingerprints:
            continue
        seen_fingerprints.add(normalized_surface)
        output.append({
            "article_id": article.article_id,
            "article_sha256": article_hash,
            "sentence_index": sentence_index,
            "source_text": sentence,
            "input": reading,
            "references": [sentence],
            "canonical_reference": sentence,
            "gold_tokens": tokens,
            "span_tokens": sum(not token["is_literal"] for token in tokens),
            "reading_length": len(reading),
            "input_length": len(reading),
            "selection_stratum": quota_stratum(sum(not token["is_literal"] for token in tokens)),
            "reading_stratum": _stratum(len(reading)),
            "normalized_surface": normalized_surface,
        })
    return output


def _load_articles(root: Path) -> Iterator[Article]:
    seen: set[str] = set()
    for path in sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.json")):
        if path.name == "manifest.json" or path.stem.lower() in {"train", "dev", "test"}:
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else None
            iterator: Iterable[Any] = rows if isinstance(rows, list) else path.read_text(encoding="utf-8").splitlines()
            for raw in iterator:
                row = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(row, Mapping) or "id" not in row or not isinstance(row.get("text"), str):
                    continue
                article_id = str(row["id"])
                if article_id not in seen:
                    seen.add(article_id)
                    yield Article(article_id, str(row["text"]), str(row.get("title", "")), str(row.get("url", "")))
        except (OSError, json.JSONDecodeError):
            continue


def _case(candidate: Mapping[str, Any], revision: str, license_name: str, number: int) -> dict[str, Any]:
    tokens = list(candidate["gold_tokens"])
    paths = list(candidate.get("gold_paths", []))
    if paths:
        path = dict(paths[0])
        path.setdefault("path_id", "canonical")
        path.setdefault("is_canonical", True)
    else:
        path = {"path_id": "canonical", "is_canonical": True, "tokens": tokens}
    empty_hash = hashlib.sha256(b"").hexdigest()
    return {
        "schema_version": "e0-case-v1", "id": f"e0-wikipedia-core-{number:04d}",
        "source": "wikipedia", "license": license_name,
        "article_id": candidate["article_id"], "article_sha256": candidate["article_sha256"],
        "sentence_index": candidate["sentence_index"], "source_revision": revision,
        "source_text_sha256": hashlib.sha256(str(candidate["source_text"]).encode("utf-8")).hexdigest(),
        "input": candidate["input"], "references": candidate["references"],
        "canonical_reference": candidate["canonical_reference"], "gold_paths": [path],
        "gold_reachable": bool(candidate.get("gold_reachable", True)),
        "span_tokens": candidate["span_tokens"], "reading_length": candidate["reading_length"],
        "input_length": candidate["input_length"], "selection_stratum": candidate["selection_stratum"],
        "selection_probability": candidate["selection_probability"], "annotation_status": "auto_extracted",
        "annotator_count": 0, "d1_anchor_ids": [],
        "normalized_fingerprint": document_fingerprint(str(candidate["source_text"])),
        "annotation_provenance": {
            "provider": "fugashi-ipadic", "model_snapshot": "ipadic-system-dictionary",
            "prompt_sha256": empty_hash, "schema_sha256": empty_hash,
            "request_parameters": {"temperature": 0, "seed": None, "top_p": 1},
            "raw_response_sha256": empty_hash, "response_id": "",
            "checker": {"provider": "", "model_snapshot": "", "raw_response_sha256": empty_hash, "status": "not_run"},
            "human_audit": [], "adjudication": {"required": True, "status": "open", "history": []},
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-manifest", type=Path, required=True)
    parser.add_argument("--d1-articles-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-count", type=int, default=2400)
    parser.add_argument("--min-articles", type=int, default=400)
    parser.add_argument("--dictionary", type=Path, default=None,
                        help="dictionary TSV used to validate provisional gold")
    parser.add_argument("--check-quota", action="store_true")
    parser.add_argument("--fill-quota", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.d1_manifest.read_text(encoding="utf-8"))
    source = manifest["source"]
    d1_index = build_e0_leakage_index(args.d1_articles_dir)
    used_ids = {article_id for split in d1_index["splits"].values() for article_id in split["article_ids"]}
    try:
        from fugashi import GenericTagger
        import ipadic
        tagger = GenericTagger(f"-r /dev/null -d {ipadic.DICDIR}")
    except ImportError as error:
        raise RuntimeError("fugashi and ipadic are required for provisional MeCab readings") from error
    pool: list[dict[str, Any]] = []
    for article in _load_articles(args.d1_articles_dir):
        if article.article_id in used_ids:
            continue
        pool.extend(extract_article_candidates(article, tagger))
    if args.dictionary is not None:
        from beme.dictionary import Dictionary
        from scripts.e0_gold_validator import validate_gold_paths
        pool = validate_gold_paths(pool, Dictionary.from_tsv(args.dictionary))
    if args.check_quota or args.fill_quota:
        from scripts.e0_quota_validator import check_quota
        quota_report = check_quota(pool)
        if args.check_quota:
            print(json.dumps(quota_report, ensure_ascii=False, sort_keys=True))
        if args.check_quota and not args.fill_quota and not quota_report["passed"]:
            raise RuntimeError("candidate pool does not satisfy E0 quota")
    findings = check_e0_leakage(pool, d1_index)
    leaked = [item for item in findings if item["leakage_detected"]]
    if leaked:
        raise RuntimeError(f"leakage detected before selection: {len(leaked)} candidates")
    selected = select_candidates(pool, args.target_count, args.seed, args.min_articles, fill_quota=args.fill_quota)
    findings = check_e0_leakage(selected, d1_index)
    if any(item["leakage_detected"] for item in findings):
        raise RuntimeError("leakage detected after selection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for number, candidate in enumerate(selected, 1):
            handle.write(json.dumps(_case(candidate, source["revision"], source["license"], number), ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
