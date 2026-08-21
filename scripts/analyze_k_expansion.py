#!/usr/bin/env python3
"""Aggregate saved AJIMEE k-expansion results by sentence span count."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


RESULT_PATTERN = re.compile(r"results_k(\d+)\.json$")


def _normalize_kana(text: str) -> str:
    return "".join(
        chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char
        for char in text
    )


def _span_count(input_text: str) -> int:
    """Count the converter's independent contiguous hiragana spans."""
    normalized = _normalize_kana(input_text)
    return sum(
        1
        for match in re.finditer(r"[\u3041-\u3096]+", normalized)
        if match.group()
    )


def _case_metrics(case: dict[str, Any], span_count: int) -> dict[str, Any]:
    return {
        "id": case["id"],
        "span_count": span_count,
        "span_class": "single-span" if span_count == 1 else "multi-span",
        "hypothesis": case["hypothesis"],
        "references": case["references"],
        "exact_match": case["exact_match"],
        "cer": case["cer"],
        "edit_distance": case["edit_distance"],
    }


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in ("single-span", "multi-span"):
        selected = [case for case in cases if case["span_class"] == label]
        count = len(selected)
        result[label] = {
            "case_count": count,
            "exact_match_count": sum(case["exact_match"] for case in selected),
            "exact_match_rate": (
                sum(case["exact_match"] for case in selected) / count if count else 0.0
            ),
            "mean_cer": sum(case["cer"] for case in selected) / count if count else 0.0,
            "mean_edit_distance": (
                sum(case["edit_distance"] for case in selected) / count if count else 0.0
            ),
        }
    return result


def _load_result(path: Path) -> tuple[int, dict[str, Any]]:
    match = RESULT_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"result filename must match results_kN.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        _case_metrics(case, _span_count(case["input"]))
        for case in payload["per_case"]
    ]
    payload["span_aggregate"] = _aggregate(cases)
    payload["span_case_count_distribution"] = {
        str(span_count): sum(case["span_count"] == span_count for case in cases)
        for span_count in sorted({case["span_count"] for case in cases})
    }
    payload["analyzed_cases"] = cases
    return int(match.group(1)), payload


def _changes(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    before = {case["id"]: case for case in baseline["analyzed_cases"]}
    after = {case["id"]: case for case in current["analyzed_cases"]}
    changed = [case_id for case_id in before if before[case_id]["hypothesis"] != after[case_id]["hypothesis"]]
    promoted = [
        case_id
        for case_id in before
        if not before[case_id]["exact_match"] and after[case_id]["exact_match"]
    ]
    regressed = [
        case_id
        for case_id in before
        if before[case_id]["exact_match"] and not after[case_id]["exact_match"]
    ]
    return {
        "changed_hypothesis_count": len(changed),
        "changed_hypothesis_ids": changed,
        "promoted_to_exact_count": len(promoted),
        "promoted_to_exact_ids": promoted,
        "regressed_from_exact_count": len(regressed),
        "regressed_from_exact_ids": regressed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="saved results_kN.json files")
    parser.add_argument("--output", type=Path, help="write the aggregate JSON here")
    args = parser.parse_args()
    paths = args.paths or sorted(Path.cwd().glob("results_k*.json"))
    if not paths:
        parser.error("no result files found")

    loaded = dict(_load_result(path) for path in paths)
    baseline_k = min(loaded)
    report = {
        "span_definition": "normalized input's contiguous hiragana runs; one run is single-span, otherwise multi-span",
        "conditions": {
            str(k): {
                "k": k,
                "candidate_latency_ms": payload.get("candidate_latency_ms"),
                "mean_cer": payload["mean_cer"],
                "exact_match_accuracy": payload["exact_match_accuracy"],
                "mean_edit_distance": payload["mean_edit_distance"],
                "span_aggregate": payload["span_aggregate"],
                "span_case_count_distribution": payload["span_case_count_distribution"],
            }
            for k, payload in sorted(loaded.items())
        },
        "changes_from_baseline": {
            str(k): _changes(loaded[baseline_k], payload)
            for k, payload in sorted(loaded.items())
            if k != baseline_k
        },
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
