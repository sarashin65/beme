"""Shared ranking aggregates and paired statistical decisions.

The functions in this module deliberately operate on per-example and
per-seed records.  In particular, they never intersect example ids or pool
contingency tables across seeds.
"""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Any, Mapping, Sequence


def _boolean_records(records: Sequence[Mapping[str, Any]], name: str) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for record in records:
        identifier = record.get("id")
        correct = record.get("correct")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{name} records must have non-empty string ids")
        if identifier in result:
            raise ValueError(f"{name} contains duplicate id: {identifier!r}")
        if type(correct) is not bool:
            raise ValueError(f"{name} record {identifier!r} must have a boolean 'correct'")
        result[identifier] = correct
    return result


def paired_table(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Pair correctness by id and return the complete 2x2 table.

    ``b`` is a loss (before only correct), while ``c`` is a gain (after only
    correct).  Refusing unequal id sets prevents a comparison from silently
    changing its population.
    """

    before_by_id = _boolean_records(before, "before")
    after_by_id = _boolean_records(after, "after")
    if before_by_id.keys() != after_by_id.keys():
        missing_after = sorted(before_by_id.keys() - after_by_id.keys())
        missing_before = sorted(after_by_id.keys() - before_by_id.keys())
        raise ValueError(
            "before and after id sets differ "
            f"(missing_after={missing_after!r}, missing_before={missing_before!r})"
        )

    table = {"a": 0, "b": 0, "c": 0, "d": 0}
    for identifier, before_correct in before_by_id.items():
        after_correct = after_by_id[identifier]
        key = (
            "a" if before_correct and after_correct else
            "b" if before_correct else
            "c" if after_correct else
            "d"
        )
        table[key] += 1
    return table


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def mcnemar(b: int, c: int) -> dict[str, Any]:
    """Return Phase 3's exact/continuity-corrected McNemar result."""

    b = _count(b, "b")
    c = _count(c, "c")
    discordant = b + c
    if discordant < 25:
        tail = sum(math.comb(discordant, i) for i in range(min(b, c) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
        method = "exact_binomial_two_sided"
    else:
        statistic = (abs(b - c) - 1.0) ** 2 / discordant if discordant else 0.0
        p_value = math.erfc(math.sqrt(statistic / 2.0))
        method = "continuity_corrected_chi_squared"
    return {"b": b, "c": c, "discordant": discordant, "method": method, "p_value": p_value}


def label(tables: Sequence[Mapping[str, Any]], *, alpha: float = 0.05) -> str:
    """Label a reproducible direction without pooling tables across seeds."""

    if isinstance(tables, Mapping):
        raise TypeError("tables must be a sequence containing one table per seed; do not pool seeds")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    materialized = list(tables)
    if len(materialized) < 2:
        return "indeterminate"

    results = [mcnemar(table.get("b"), table.get("c")) for table in materialized]
    if any(result["p_value"] >= alpha for result in results):
        return "indeterminate"
    directions = [
        1 if result["c"] > result["b"] else -1 if result["b"] > result["c"] else 0
        for result in results
    ]
    median_direction = statistics.median(result["c"] - result["b"] for result in results)
    if all(direction == 1 for direction in directions) and median_direction > 0:
        return "significant_better"
    if all(direction == -1 for direction in directions) and median_direction < 0:
        return "significant_worse"
    return "indeterminate"


def ranked_candidates_from_scores(
    candidates: Sequence[Mapping[str, Any]],
    *,
    viterbi_weight: float,
    reranker_weight: float,
    rerank_k: int | None,
) -> list[dict[str, Any]]:
    """Reproduce Converter's reranked head plus untouched tail ordering."""

    materialized = [dict(candidate) for candidate in candidates]
    if rerank_k is not None and (type(rerank_k) is not int or rerank_k <= 0):
        raise ValueError("rerank_k must be a positive integer or None")
    for candidate in materialized:
        if not isinstance(candidate.get("surface"), str):
            raise ValueError("each candidate must have a string surface")
        try:
            cost = float(candidate["cost"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each candidate must have a numeric cost") from exc
        if not math.isfinite(cost):
            raise ValueError("candidate costs must be finite")
    # Empty context is encoded by null relevance.  A partially scored record is
    # unsafe too, so any null keeps the complete original order.
    if not materialized or any(candidate.get("relevance") is None for candidate in materialized):
        return materialized

    head = materialized if rerank_k is None else materialized[:rerank_k]
    try:
        head = sorted(
            head,
            key=lambda candidate: (
                float(viterbi_weight) * float(candidate["cost"])
                - float(reranker_weight) * float(candidate["relevance"]),
                float(candidate["cost"]),
                candidate["surface"],
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate relevance and weights must be numeric") from exc
    tail = [] if rerank_k is None else materialized[rerank_k:]
    return head + tail


def top1_from_scores(
    candidates: Sequence[Mapping[str, Any]],
    *,
    viterbi_weight: float,
    reranker_weight: float,
    rerank_k: int | None,
) -> str:
    """Return the top surface under Converter's deterministic sorting key."""

    ranked = ranked_candidates_from_scores(
        candidates,
        viterbi_weight=viterbi_weight,
        reranker_weight=reranker_weight,
        rerank_k=rerank_k,
    )
    if not ranked:
        raise ValueError("candidates must not be empty")
    return ranked[0]["surface"]


_FIXED_BUCKETS = {
    "contrast_pair": ("false", "true"),
    # "17+" is an overflow bucket: real readings are not capped at 16
    # characters, and the fixed buckets below it must not raise just because
    # a longer one showed up (issue #9 Phase 4 review item 7).
    "reading_length": ("1-3", "4-6", "7-10", "11-16", "17+"),
    "proper_noun": ("false", "true"),
}


def _stratum(record: Mapping[str, Any], axis: str) -> str:
    if axis == "contrast_pair":
        return "true" if record.get("contrast_pair_id") is not None else "false"
    if axis == "reading_length":
        length = record.get("reading_length")
        if length is None and isinstance(record.get("reading"), str):
            length = len(record["reading"])
        if type(length) is not int:
            raise ValueError("reading_length records need an integer reading_length or string reading")
        for bucket, low, high in (("1-3", 1, 3), ("4-6", 4, 6), ("7-10", 7, 10), ("11-16", 11, 16)):
            if low <= length <= high:
                return bucket
        if length >= 17:
            return "17+"
        raise ValueError("reading_length must be a positive integer")
    if axis == "span_tokens":
        value = record.get("span_tokens")
        if value is None:
            return "unknown"
        if type(value) is not int or value <= 0:
            raise ValueError("span_tokens must be a positive integer or None")
        return str(value)
    if axis == "proper_noun":
        value = record.get("proper_noun")
        if type(value) is not bool:
            raise ValueError("proper_noun records need a boolean proper_noun")
        return "true" if value else "false"
    raise ValueError(f"unknown stratification axis: {axis}")


def stratify(records: Sequence[Mapping[str, Any]], axis: str) -> dict[str, dict[str, Any]]:
    """Aggregate before/after error rates for one documented axis."""

    if axis not in {*_FIXED_BUCKETS, "span_tokens"}:
        raise ValueError(f"unknown stratification axis: {axis}")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if type(record.get("before_correct")) is not bool or type(record.get("after_correct")) is not bool:
            raise ValueError("stratification records need boolean before_correct and after_correct")
        groups[_stratum(record, axis)].append(record)
    labels = list(_FIXED_BUCKETS.get(axis, ()))
    labels.extend(label for label in groups if label not in labels)
    if axis == "span_tokens":
        labels.sort(key=lambda value: (value == "unknown", int(value) if value != "unknown" else 0))
    result: dict[str, dict[str, Any]] = {}
    for layer in labels:
        group = groups[layer]
        n = len(group)
        result[layer] = {
            "n": n,
            "before_error_rate": (sum(not item["before_correct"] for item in group) / n if n else None),
            "after_error_rate": (sum(not item["after_correct"] for item in group) / n if n else None),
            "low_support": n < 30,
        }
    return result


def pair_level(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count complete contrast pairs; exclude incomplete groups from outcomes."""

    groups: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        pair_id = record.get("contrast_pair_id")
        if pair_id is None:
            continue
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("contrast_pair_id must be a non-empty string or None")
        correct = record.get("correct")
        if type(correct) is not bool:
            raise ValueError("pair records need a boolean correct")
        groups[pair_id].append(correct)
    result = {"both_correct": 0, "one_correct": 0, "both_wrong": 0, "incomplete_pairs": 0}
    for correctness in groups.values():
        if len(correctness) != 2:
            result["incomplete_pairs"] += 1
        elif all(correctness):
            result["both_correct"] += 1
        elif any(correctness):
            result["one_correct"] += 1
        else:
            result["both_wrong"] += 1
    return result


def _gold_rank(record: Mapping[str, Any]) -> int | None:
    gold = record.get("gold")
    candidates = record.get("candidates")
    if not isinstance(gold, str) or not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("rank records need a string gold and a candidate sequence")
    for index, candidate in enumerate(candidates, 1):
        surface = candidate.get("surface") if isinstance(candidate, Mapping) else candidate
        if surface == gold:
            return index
    return None


def gold_rank_histogram(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count every record, including gold surfaces absent from candidates."""

    result = {str(index): 0 for index in range(1, 21)} | {">20": 0, "absent": 0}
    for record in records:
        rank = _gold_rank(record)
        if rank is None:
            result["absent"] += 1
        elif rank <= 20:
            result[str(rank)] += 1
        else:
            result[">20"] += 1
    return result


__all__ = [
    "gold_rank_histogram",
    "label",
    "mcnemar",
    "paired_table",
    "pair_level",
    "ranked_candidates_from_scores",
    "stratify",
    "top1_from_scores",
]
