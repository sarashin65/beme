"""Strict JSONL contracts and reproducible manifests for scorer training data."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


_ALLOWED_KEYS = frozenset({
    "id", "source", "article_id", "left_context", "right_context",
    "reading", "target_surface", "positive", "candidates", "viterbi_costs",
    "frequency_bucket",
    "article_sha256", "context_truncated", "contrast_pair_id", "gold_index",
    "span_tokens", "split",
})
_SPLIT_NAMES = ("train", "dev", "test")


class _DuplicateKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One labeled example; contexts are provenance, not local-scorer input."""

    reading: str
    target_surface: str
    id: str | None = None
    source: str | None = None
    article_id: str | None = None
    left_context: str = ""
    right_context: str = ""
    candidates: tuple[str, ...] = ()
    viterbi_costs: tuple[float, ...] = ()
    frequency_bucket: Mapping[str, str] = MappingProxyType({})
    article_sha256: str | None = None
    context_truncated: bool | None = None
    contrast_pair_id: str | None = None
    gold_index: int | None = None
    span_tokens: tuple[Any, ...] = ()
    split: str | None = None


def _line_error(path: Path, line: int, message: str) -> ValueError:
    return ValueError(f"{path}: line {line}: {message}")


def _string(value: Any, name: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        requirement = "non-empty string" if nonempty else "string"
        raise ValueError(f"{name} must be a {requirement}")
    return value


def _optional_string(value: dict[str, Any], name: str) -> str | None:
    if name not in value:
        return None
    if not isinstance(value[name], str):
        raise ValueError(f"{name} must be a string")
    return value[name]


def _optional_nullable_string(value: dict[str, Any], name: str) -> str | None:
    if name not in value or value[name] is None:
        return None
    return _string(value[name], name, nonempty=True)


def _parse_example(value: Any) -> TrainingExample:
    if not isinstance(value, dict):
        raise ValueError("JSON value must be an object")
    unknown = sorted(set(value) - _ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"unknown field: {unknown[0]!r}")

    reading = _string(value.get("reading"), "reading", nonempty=True)
    target = (
        _string(value["target_surface"], "target_surface", nonempty=True)
        if "target_surface" in value else None
    )
    positive = (
        _string(value["positive"], "positive", nonempty=True)
        if "positive" in value else None
    )
    if target is None and positive is None:
        raise ValueError("one of 'target_surface' or 'positive' is required")
    if target is not None and positive is not None and target != positive:
        raise ValueError("'target_surface' and 'positive' must be equal")
    target = target if target is not None else positive
    assert target is not None

    candidates = value.get("candidates", [])
    if not isinstance(candidates, list) or any(not isinstance(item, str) for item in candidates):
        raise ValueError("candidates must be an array of strings")
    costs = value.get("viterbi_costs", [])
    if (
        not isinstance(costs, list)
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in costs)
    ):
        raise ValueError("viterbi_costs must be an array of numbers")
    if len(candidates) != len(costs):
        raise ValueError("candidates and viterbi_costs must have equal lengths")

    article_sha256 = value.get("article_sha256")
    if "article_sha256" in value and (
        not isinstance(article_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", article_sha256) is None
    ):
        raise ValueError("article_sha256 must be a 64-character hexadecimal string")
    context_truncated = value.get("context_truncated")
    if "context_truncated" in value and type(context_truncated) is not bool:
        raise ValueError("context_truncated must be a boolean")

    span_tokens = value.get("span_tokens", [])
    if not isinstance(span_tokens, list):
        raise ValueError("span_tokens must be an array")
    split = value.get("split")
    if "split" in value and (not isinstance(split, str) or split not in _SPLIT_NAMES):
        raise ValueError("split must be one of train, dev, or test")

    gold_index = value.get("gold_index")
    if "gold_index" in value:
        if isinstance(gold_index, bool) or not isinstance(gold_index, int) or gold_index < 0:
            raise ValueError("gold_index must be a non-negative integer")
        if not candidates:
            raise ValueError("gold_index requires candidates")
        if gold_index >= len(candidates):
            raise ValueError("gold_index is outside candidates")
        if candidates[gold_index] != target:
            raise ValueError("candidates[gold_index] must equal target_surface")

    buckets = value.get("frequency_bucket", {})
    if not isinstance(buckets, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in buckets.items()
    ):
        raise ValueError("frequency_bucket must be an object of strings")

    return TrainingExample(
        reading=reading,
        target_surface=target,
        id=_optional_string(value, "id"),
        source=_optional_string(value, "source"),
        article_id=_optional_string(value, "article_id"),
        left_context=_string(value.get("left_context", ""), "left_context"),
        right_context=_string(value.get("right_context", ""), "right_context"),
        candidates=tuple(candidates),
        viterbi_costs=tuple(float(cost) for cost in costs),
        frequency_bucket=MappingProxyType(dict(sorted(buckets.items()))),
        article_sha256=article_sha256,
        context_truncated=context_truncated,
        contrast_pair_id=_optional_nullable_string(value, "contrast_pair_id"),
        gold_index=gold_index,
        span_tokens=tuple(span_tokens),
        split=split,
    )


def load_jsonl(path: str | Path) -> tuple[TrainingExample, ...]:
    """Load one strict UTF-8 JSONL split, preserving input order."""

    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        line = raw[:error.start].count(b"\n") + 1
        raise _line_error(path, line, "file is not valid UTF-8") from error

    examples: list[TrainingExample] = []
    ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise _line_error(path, line_number, "blank line is not allowed")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
            example = _parse_example(value)
        except (json.JSONDecodeError, _DuplicateKey, TypeError, ValueError) as error:
            raise _line_error(path, line_number, str(error)) from error
        if example.id is not None and example.id in ids:
            raise _line_error(path, line_number, f"duplicate id: {example.id!r}")
        if example.id is not None:
            ids.add(example.id)
        examples.append(example)
    return tuple(examples)


load_training_examples = load_jsonl


def _path_string(path: str | Path) -> str:
    return str(path)


def build_manifest(
    paths: Mapping[str, str | Path],
    splits: Mapping[str, tuple[TrainingExample, ...]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic manifest from split paths and loaded examples."""

    if set(paths) != set(_SPLIT_NAMES):
        raise ValueError("paths must contain exactly train, dev, and test")
    loaded = dict(splits) if splits is not None else {
        name: load_jsonl(paths[name]) for name in _SPLIT_NAMES
    }
    result: dict[str, Any] = {"splits": {}}
    for name in sorted(_SPLIT_NAMES):
        path = Path(paths[name])
        examples = loaded[name]
        result["splits"][name] = {
            "path": _path_string(paths[name]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "examples": len(examples),
            "count": len(examples),
            "source_counts": dict(sorted(Counter(
                example.source for example in examples if example.source is not None
            ).items())),
            "article_disjoint_verified": False,
            "context_truncated_count": sum(
                example.context_truncated is not None for example in examples
            ),
            "contrast_pair_count": sum(
                example.contrast_pair_id is not None for example in examples
            ),
            "article_sha256_present_count": sum(
                example.article_sha256 is not None for example in examples
            ),
            "span_tokens_present_count": sum(
                bool(example.span_tokens) for example in examples
            ),
            "split_field_verified": bool(examples) and all(
                example.split is not None and example.split == name for example in examples
            ),
        }

    article_sets = {
        name: {example.article_id for example in loaded[name]}
        for name in _SPLIT_NAMES
    }
    complete = bool(any(loaded.values())) and all(
        example.article_id is not None for examples in loaded.values() for example in examples
    )
    disjoint = complete and all(
        article_sets[left].isdisjoint(article_sets[right])
        for index, left in enumerate(_SPLIT_NAMES)
        for right in _SPLIT_NAMES[index + 1:]
    )
    article_hashes: dict[str, str] = {}
    article_sha256_consistent = True
    for examples in loaded.values():
        for example in examples:
            if example.article_id is None or example.article_sha256 is None:
                continue
            previous = article_hashes.setdefault(example.article_id, example.article_sha256)
            if previous != example.article_sha256:
                article_sha256_consistent = False
    contrast_pairs: dict[str, set[str]] = {}
    for name, examples in loaded.items():
        for example in examples:
            if example.contrast_pair_id is not None:
                contrast_pairs.setdefault(example.contrast_pair_id, set()).add(name)
    contrast_pairs_disjoint = all(len(split_names) == 1 for split_names in contrast_pairs.values())
    split_field_verified = bool(any(loaded.values())) and all(
        example.split is not None and example.split == name
        for name, examples in loaded.items() for example in examples
    )
    for split in result["splits"].values():
        split["article_disjoint_verified"] = disjoint
    result["article_sha256_consistent"] = article_sha256_consistent
    result["contrast_pairs_disjoint"] = contrast_pairs_disjoint
    result["split_field_verified"] = split_field_verified
    result["article_disjoint_verified"] = disjoint
    return result


def load_splits(
    paths: Mapping[str, str | Path],
) -> tuple[dict[str, tuple[TrainingExample, ...]], dict[str, Any]]:
    """Load train/dev/test and reject IDs or article IDs crossing splits."""

    if set(paths) != set(_SPLIT_NAMES):
        raise ValueError("paths must contain exactly train, dev, and test")
    splits = {name: load_jsonl(paths[name]) for name in _SPLIT_NAMES}
    ids: dict[str, str] = {}
    for split, examples in splits.items():
        for example in examples:
            if example.id is not None and example.id in ids:
                raise ValueError(
                    f"split {split}: duplicate id {example.id!r} also occurs in {ids[example.id]}"
                )
            if example.id is not None:
                ids[example.id] = split

    article_sets = {
        name: {example.article_id for example in examples if example.article_id is not None}
        for name, examples in splits.items()
    }
    article_hashes: dict[str, str] = {}
    for split, examples in splits.items():
        for example in examples:
            if example.split is not None and example.split != split:
                raise ValueError(
                    f"split field {example.split!r} does not match {split} input"
                )
            if example.article_id is not None and example.article_sha256 is not None:
                previous = article_hashes.setdefault(example.article_id, example.article_sha256)
                if previous != example.article_sha256:
                    raise ValueError(
                        f"article_id {example.article_id!r} has inconsistent article_sha256"
                    )
    contrast_locations: dict[str, str] = {}
    for split, examples in splits.items():
        for example in examples:
            if example.contrast_pair_id is not None:
                previous = contrast_locations.setdefault(example.contrast_pair_id, split)
                if previous != split:
                    raise ValueError(
                        f"contrast_pair_id {example.contrast_pair_id!r} crosses splits"
                    )
    for index, left in enumerate(_SPLIT_NAMES):
        for right in _SPLIT_NAMES[index + 1:]:
            overlap = article_sets[left] & article_sets[right]
            if overlap:
                article = sorted(overlap)[0]
                raise ValueError(f"article_id {article!r} crosses {left}/{right} splits")
    return splits, build_manifest(paths, splits)


load_training_splits = load_splits


__all__ = [
    "TrainingExample", "build_manifest", "load_jsonl", "load_splits",
    "load_training_examples", "load_training_splits",
]
