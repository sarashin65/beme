"""Quota checks for the E0 sentence-level selection pool."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

QUOTA = (("2-4", 2, 4, 0.10, 200), ("5-8", 5, 8, 0.25, 500),
         ("9-16", 9, 16, 0.40, 800), ("17+", 17, None, 0.25, 500))


def quota_stratum(span_tokens: int) -> str:
    for label, lower, upper, _ratio, _minimum in QUOTA:
        if upper is None or lower <= span_tokens <= upper:
            return label
    return "<2"


def _counts(candidates: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(quota_stratum(int(candidate.get("span_tokens", 0))) for candidate in candidates)


def check_quota(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(candidates)
    counts = _counts(candidates)
    strata: dict[str, dict[str, Any]] = {}
    for label, _lower, _upper, ratio, minimum in QUOTA:
        count = counts[label]
        strata[label] = {
            "count": count, "ratio": count / total if total else 0.0,
            "target_ratio": ratio, "minimum": minimum,
            "minimum_required": max(minimum, math.ceil(total * ratio)),
            "meets_ratio": count / total >= ratio if total else False,
            "meets_minimum": count >= minimum,
        }
    gate_values = {
        "span_tokens_ge_5": sum(int(c.get("span_tokens", 0)) >= 5 for c in candidates),
        "reading_length_ge_17": sum(int(c.get("reading_length", 0)) >= 17 for c in candidates),
        "reading_length_ge_33": sum(int(c.get("reading_length", 0)) >= 33 for c in candidates),
    }
    gates = {
        key: {"count": value, "ratio": value / total if total else 0.0,
              "target_ratio": target, "passed": value / total >= target if total else False}
        for (key, value, target) in (("span_tokens_ge_5", gate_values["span_tokens_ge_5"], 0.90),
                                     ("reading_length_ge_17", gate_values["reading_length_ge_17"], 0.70),
                                     ("reading_length_ge_33", gate_values["reading_length_ge_33"], 0.25))}
    return {"total": total, "counts": dict(counts), "strata": strata, "gates": gates,
            "passed": all(s["meets_ratio"] and s["meets_minimum"] for s in strata.values()) and all(g["passed"] for g in gates.values())}


def quota_gap(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    report = check_quota(candidates)
    return {label: max(0, values["minimum_required"] - values["count"])
            for label, values in report["strata"].items() if values["minimum_required"] > values["count"]}


def suggest_quota_adjustment(candidates: Sequence[Mapping[str, Any]], target_count: int) -> dict[str, Any]:
    if target_count < 1:
        raise ValueError("target_count must be positive")
    counts = _counts(candidates)
    desired = {label: math.ceil(target_count * ratio) for label, _l, _u, ratio, _m in QUOTA}
    gaps = {label: max(0, desired[label] - counts[label]) for label in desired}
    return {"target_count": target_count, "current_counts": dict(counts), "desired_counts": desired,
            "needed": gaps, "priority": sorted(gaps, key=lambda label: (-gaps[label], label))}


__all__ = ["check_quota", "quota_gap", "quota_stratum", "suggest_quota_adjustment"]
