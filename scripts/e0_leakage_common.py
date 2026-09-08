"""Shared document normalisation and leakage-index helpers.

This module deliberately has no dependency on the d1 CLI.  It accepts plain
records as well as objects (for example ``TrainingExample``), so it can be
used by both the existing audit and the E0 candidate builder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping

NGRAM_SIZE = 12
_SPLITS = ("train", "dev", "test")


def normalize_text(value: str) -> str:
    """Return NFKC-normalised text with all Unicode whitespace removed."""
    return remove_whitespace(unicodedata.normalize("NFKC", value))


def remove_whitespace(value: str) -> str:
    """Remove Unicode whitespace without changing any other characters."""
    return re.sub(r"\s+", "", value)


def document_ngrams(value: str, size: int = NGRAM_SIZE) -> set[str]:
    """Return the set of normalised, contiguous n-grams in *value*."""
    if size < 1:
        raise ValueError("n-gram size must be positive")
    text = normalize_text(value)
    return {text[i : i + size] for i in range(max(0, len(text) - size + 1))}


def ngram_fingerprint(value: str, size: int = NGRAM_SIZE) -> frozenset[str]:
    """Return a hashable 12-gram fingerprint (or a requested n-gram size)."""
    return frozenset(document_ngrams(value, size))


def document_fingerprint(value: str) -> str:
    """Return the SHA-256 fingerprint of normalised document text."""
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def _get(record: Any, name: str, default: Any = "") -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _record_text(record: Any) -> str:
    for name in ("text", "article_text", "source_text", "source_sentence", "sentence", "document"):
        value = _get(record, name, None)
        if isinstance(value, str) and value:
            return value
    references = _get(record, "references", None)
    if isinstance(references, (list, tuple)):
        return "".join(str(value) for value in references)
    # d1 has no full article field; this is the maximum available document
    # material and matches the legacy audit's context components.
    return "".join(str(_get(record, name, "")) for name in ("left_context", "target_surface", "positive", "right_context"))


def _iter_records(path: Path) -> Iterable[tuple[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield path.stem, json.loads(line)
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            yield from ((path.stem, item) for item in value)
        else:
            yield path.stem, value
    elif path.suffix.lower() in {".txt", ".text"}:
        yield path.stem, {"text": path.read_text(encoding="utf-8")}


def build_e0_leakage_index(articles_dir: str | Path) -> dict[str, Any]:
    """Build a deterministic forbidden-document index from *articles_dir*.

    Files named ``train``, ``dev`` and ``test`` (with any supported extension)
    are treated as d1 split records.  Other files are treated as article
    records and placed in the ``articles`` bucket.  The returned structure is
    JSON-serialisable and includes article IDs, exact normalised fingerprints,
    and 12-gram postings for each split.
    """
    root = Path(articles_dir)
    paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jsonl", ".json", ".txt", ".text"})
    index: dict[str, Any] = {"ngram_size": NGRAM_SIZE, "splits": {}, "articles": {}}
    for split in _SPLITS:
        index["splits"][split] = {"article_ids": [], "fingerprints": [], "ngrams": {}, "documents": {}}

    for path in paths:
        stem = path.stem.lower()
        split = next((name for name in _SPLITS if stem == name or stem.startswith(name + "_")), None)
        for fallback_id, record in _iter_records(path):
            article_id = _get(record, "article_id", None) or _get(record, "id", None) or fallback_id
            article_id = str(article_id)
            text = _record_text(record)
            fingerprint = document_fingerprint(text)
            grams = document_ngrams(text)
            record_split = _get(record, "split", None)
            effective_split = record_split if record_split in _SPLITS else split
            bucket = index["splits"][effective_split] if effective_split else index["articles"]
            bucket["article_ids"] = sorted(set(bucket.get("article_ids", [])) | {article_id})
            bucket["fingerprints"] = sorted(set(bucket.get("fingerprints", [])) | {fingerprint})
            bucket.setdefault("documents", {})[article_id] = {
                "fingerprint": fingerprint,
                "ngrams": sorted(grams),
            }
            postings = bucket.setdefault("ngrams", {})
            for gram in grams:
                postings.setdefault(gram, [])
                if article_id not in postings[gram]:
                    postings[gram].append(article_id)
                    postings[gram].sort()
    return index


def check_e0_leakage(candidates: Iterable[Any], index: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return candidate-level blocking overlaps against d1 train/dev/test.

    Each result has ``case_id`` and an ``overlaps`` list.  Overlap types are
    ``article_id``, ``document`` (exact normalised text), and ``12gram``.
    Candidates without an overlap are included with an empty list so callers
    can retain a complete audit trail.
    """
    results = []
    splits = index.get("splits", {})
    for position, candidate in enumerate(candidates):
        case_id = str(_get(candidate, "case_id", None) or _get(candidate, "id", None) or position)
        article_id = _get(candidate, "article_id", None)
        text = _record_text(candidate)
        fingerprint = document_fingerprint(text)
        grams = document_ngrams(text, int(index.get("ngram_size", NGRAM_SIZE)))
        overlaps: list[dict[str, Any]] = []
        for split in _SPLITS:
            bucket = splits.get(split, {})
            if article_id is not None and str(article_id) in set(bucket.get("article_ids", [])):
                overlaps.append({"split": split, "type": "article_id", "article_id": str(article_id)})
            if fingerprint in set(bucket.get("fingerprints", [])):
                overlaps.append({"split": split, "type": "document", "fingerprint": fingerprint})
            # Flatten postings while retaining the matching article IDs.
            matched_ids = sorted({article for gram in grams for article in bucket.get("ngrams", {}).get(gram, [])})
            if matched_ids:
                overlaps.append({"split": split, "type": "12gram", "article_ids": matched_ids, "count": len(matched_ids)})
        results.append({"case_id": case_id, "overlaps": overlaps, "leakage_detected": bool(overlaps)})
    return results


# Compatibility names used by the original audit and by small downstream tools.
_normalized = normalize_text
ngrams = document_ngrams
