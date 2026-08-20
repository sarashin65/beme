"""Information-retrieval evaluation for deterministic conversion candidates."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


_REQUIRED_CASE_KEYS = frozenset({"id", "reading", "judgments"})


def _reject_json_constant(value: str) -> None:
    """Reject non-standard JSON values accepted by Python by default."""
    raise ValueError(f"non-standard JSON constant {value!r}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _line_error(line_number: int, message: str) -> ValueError:
    return ValueError(f"line {line_number}: {message}")


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load and strictly validate evaluation cases from UTF-8 JSONL.

    Every physical line must contain one JSON object with exactly ``id``,
    ``reading``, and ``judgments`` fields.  Judgment values are positive
    integers.  Validation errors identify the offending line.
    """
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    case_path = Path(path)

    try:
        handle = case_path.open("rb")
    except OSError:
        raise
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise _line_error(line_number, "invalid UTF-8") from exc
            if not line.strip():
                raise _line_error(line_number, "blank lines are not allowed")
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_object_without_duplicate_keys,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise _line_error(line_number, f"invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise _line_error(line_number, "case must be a JSON object")
            if set(value) != _REQUIRED_CASE_KEYS:
                missing = sorted(_REQUIRED_CASE_KEYS.difference(value))
                extra = sorted(set(value).difference(_REQUIRED_CASE_KEYS))
                details: list[str] = []
                if missing:
                    details.append(f"missing fields {missing}")
                if extra:
                    details.append(f"unexpected fields {extra}")
                raise _line_error(line_number, "; ".join(details))

            case_id = value["id"]
            if not isinstance(case_id, str) or not case_id.strip():
                raise _line_error(line_number, "id must be a non-empty string")
            if case_id in seen_ids:
                raise _line_error(line_number, f"duplicate id {case_id!r}")
            seen_ids.add(case_id)

            reading = value["reading"]
            if not isinstance(reading, str) or not reading.strip():
                raise _line_error(line_number, "reading must be a non-empty string")

            judgments = value["judgments"]
            if not isinstance(judgments, dict):
                raise _line_error(line_number, "judgments must be an object")
            for surface, relevance in judgments.items():
                if not isinstance(surface, str) or not surface:
                    raise _line_error(line_number, "judgment surfaces must be non-empty strings")
                if isinstance(relevance, bool) or not isinstance(relevance, int) or relevance <= 0:
                    raise _line_error(
                        line_number,
                        f"judgment for {surface!r} must be a positive integer",
                    )

            # Copy the nested mapping so callers cannot mutate the parser's
            # intermediate object through an alias.
            cases.append(
                {
                    "id": case_id,
                    "reading": reading,
                    "judgments": dict(judgments),
                }
            )
    return cases


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be an integer")
    return max(k, 0)


def _surface(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    try:
        return candidate.surface
    except AttributeError as exc:
        raise TypeError("retrieved candidates must be strings or expose .surface") from exc


def _relevance(surface: str, judgments: Mapping[str, int]) -> float:
    value = judgments.get(surface, 0)
    if isinstance(value, bool):
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if numeric > 0 else 0.0


def ndcg_at_k(
    retrieved: Sequence[Any] | Iterable[Any],
    judgments: Mapping[str, int],
    k: int,
) -> float:
    """Return standard graded NDCG at ``k``.

    Gain is ``2**relevance - 1`` and rank discount is ``log2(rank + 1)``.
    The ideal ranking is formed from the supplied judgments.  Queries with
    no judged relevant items, or an empty retrieved list, score ``0.0``.
    """
    cutoff = _validate_k(k)
    if cutoff == 0:
        return 0.0
    retrieved_list = list(retrieved)
    if not retrieved_list or not judgments:
        return 0.0

    def discounted_gain(relevances: Sequence[float]) -> float:
        return sum(
            (2.0**relevance - 1.0) / math.log2(rank + 2)
            for rank, relevance in enumerate(relevances)
        )

    observed = [
        _relevance(_surface(candidate), judgments)
        for candidate in retrieved_list[:cutoff]
    ]
    ideal = sorted(
        (_relevance(surface, judgments) for surface in judgments),
        reverse=True,
    )[:cutoff]
    ideal_dcg = discounted_gain(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return discounted_gain(observed) / ideal_dcg


def mrr_at_k(
    retrieved: Sequence[Any] | Iterable[Any],
    judgments: Mapping[str, int],
    k: int,
) -> float:
    """Return reciprocal rank of the first judged-relevant result at ``k``."""
    cutoff = _validate_k(k)
    if cutoff == 0 or not judgments:
        return 0.0
    for rank, candidate in enumerate(list(retrieved)[:cutoff], start=1):
        if _relevance(_surface(candidate), judgments) > 0:
            return 1.0 / rank
    return 0.0


def _case_value(case: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(case, Mapping):
        return case[name]
    return getattr(case, name)


def evaluate(
    converter: Any,
    cases: Iterable[Mapping[str, Any] | Any],
    k: int = 5,
    cascade_threshold: float = 2.0,
) -> dict[str, Any]:
    """Evaluate converter output and return per-query and aggregate metrics."""
    cutoff = _validate_k(k)
    per_query: list[dict[str, Any]] = []
    ndcg_total = 0.0
    mrr_total = 0.0
    top1_total = 0
    rerank_total = 0

    for case in cases:
        case_id = _case_value(case, "id")
        reading = _case_value(case, "reading")
        judgments = _case_value(case, "judgments")
        result = converter.convert(
            reading,
            k=cutoff,
            cascade_threshold=cascade_threshold,
        )
        candidates = list(result.candidates)
        surfaces = [_surface(candidate) for candidate in candidates]
        query_ndcg = ndcg_at_k(surfaces, judgments, cutoff)
        query_mrr = mrr_at_k(surfaces, judgments, cutoff)
        top1_correct = bool(surfaces and _relevance(surfaces[0], judgments) > 0)
        needs_rerank = bool(result.needs_rerank)
        ndcg_total += query_ndcg
        mrr_total += query_mrr
        top1_total += int(top1_correct)
        rerank_total += int(needs_rerank)
        per_query.append(
            {
                "id": case_id,
                "reading": reading,
                "retrieved": surfaces,
                "ndcg_at_k": query_ndcg,
                "mrr_at_k": query_mrr,
                "top1_correct": top1_correct,
                "confidence": result.confidence,
                "needs_rerank": needs_rerank,
            }
        )

    query_count = len(per_query)
    denominator = query_count or 1
    return {
        "k": cutoff,
        "query_count": query_count,
        "ndcg_at_k": ndcg_total / denominator,
        "mrr_at_k": mrr_total / denominator,
        "top1_accuracy": top1_total / denominator,
        "rerank_rate": rerank_total / denominator,
        "per_query": per_query,
    }


# A descriptive alias for callers that prefer the operation in the name.
evaluate_cases = evaluate


__all__ = ["evaluate", "evaluate_cases", "load_cases", "mrr_at_k", "ndcg_at_k"]
