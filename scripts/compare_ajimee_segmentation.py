#!/usr/bin/env python3
"""Compare AJIMEE error types for two weight files, case by case.

This is an evaluation-only script.  It deliberately reuses the same sentence
conversion and oracle-path machinery as the benchmark, without changing
production code.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beme.benchmark import (  # noqa: E402
    character_error_rate,
    convert_sentence,
    edit_distance,
    katakana_to_hiragana,
    load_ajimee_cases,
)
from beme.converter import Converter  # noqa: E402
from beme.dictionary import Dictionary  # noqa: E402
from beme.features import extractor_for  # noqa: E402
from beme.lattice import build_lattice  # noqa: E402
from beme.oracle import constrained_best_path  # noqa: E402
from beme.scorer import LinearScorer  # noqa: E402
from beme.weights import Weights  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline-weights", type=Path, required=True)
    parser.add_argument("--treatment-weights", type=Path, required=True)
    parser.add_argument("--top-k", type=_positive_int, required=True)
    parser.add_argument("--beam-width", type=_positive_int, required=True)
    parser.add_argument("--cascade-threshold", type=_nonnegative_float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def boundaries(path: Any) -> list[int]:
    """Return internal cumulative reading boundaries for a lattice path."""
    tokens = list(path.tokens)
    result: list[int] = []
    position = 0
    for token in tokens[:-1]:
        position += len(token.reading)
        result.append(position)
    return result


def _selected_reference(case: Any, hypothesis: str) -> tuple[str, int, float]:
    scored = [
        (character_error_rate(reference, hypothesis), edit_distance(reference, hypothesis), order, reference)
        for order, reference in enumerate(case.references)
    ]
    cer, distance, _order, reference = min(scored)
    return reference, distance, cer


def analyze_case(case: Any, converter: Converter, top_k: int, threshold: float) -> dict[str, Any]:
    reading = katakana_to_hiragana(case.input)
    candidates = convert_sentence(
        converter,
        case.input,
        k=top_k,
        cascade_threshold=threshold,
        preceding_context=case.context,
    )
    hypothesis = candidates[0].surface if candidates else ""
    reference, distance, cer = _selected_reference(case, hypothesis)
    exact = hypothesis in case.references
    lattice = build_lattice(reading, converter.dictionary, allow_literals=True)
    gold_path = constrained_best_path(lattice, reference, converter.scorer)
    hyp_path = constrained_best_path(lattice, hypothesis, converter.scorer) if hypothesis else None
    gold_in_lattice = gold_path is not None
    gold_bounds = boundaries(gold_path) if gold_path is not None else None
    hyp_bounds = boundaries(hyp_path) if hyp_path is not None else None

    if exact:
        error_type = "correct"
    elif not gold_in_lattice:
        error_type = "coverage"
    elif hyp_path is None or hyp_bounds != gold_bounds:
        error_type = "segmentation"
    else:
        error_type = "homophone"

    return {
        "hypothesis": hypothesis,
        "cer": cer,
        "exact_match": exact,
        "gold_in_lattice": gold_in_lattice,
        "error_type": error_type,
        "gold_boundaries": gold_bounds,
        "hypothesis_boundaries": hyp_bounds,
        "reference": reference,
        "edit_distance": distance,
    }


def summarize(records: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    errors = [record[condition] for record in records if not record[condition]["exact_match"]]
    types = Counter(record["error_type"] for record in errors)
    return {
        "condition": condition,
        "case_count": len(records),
        "errors": len(errors),
        "error_rate": len(errors) / len(records) if records else 0.0,
        "by_type": dict(sorted(types.items())),
        "gold_in_lattice": sum(record[condition]["gold_in_lattice"] for record in records),
        "mean_cer": sum(record[condition]["cer"] for record in records) / len(records) if records else 0.0,
        "exact_match": sum(record[condition]["exact_match"] for record in records) / len(records) if records else 0.0,
    }


def main() -> int:
    args = parse_args()
    dictionary = Dictionary.from_tsv(args.dictionary)
    cases = load_ajimee_cases(args.cases)

    converters: dict[str, Converter] = {}
    for label, weights_path in (("baseline", args.baseline_weights), ("treatment", args.treatment_weights)):
        weights = Weights.load(weights_path, feature_spec=None)
        extractor = extractor_for(weights.feature_spec, dictionary.reading_conditional)
        converters[label] = Converter(
            dictionary,
            beam_width=args.beam_width,
            scorer=LinearScorer(extractor, weights),
        )

    per_case: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        if index % 50 == 0:
            print(f"{index}/{len(cases)}", file=sys.stderr, flush=True)
        per_case.append({
            "id": case.id,
            "baseline": analyze_case(case, converters["baseline"], args.top_k, args.cascade_threshold),
            "treatment": analyze_case(case, converters["treatment"], args.top_k, args.cascade_threshold),
        })

    baseline_errors = sum(not item["baseline"]["exact_match"] for item in per_case)
    treatment_errors = sum(not item["treatment"]["exact_match"] for item in per_case)
    baseline_seg = {item["id"] for item in per_case if item["baseline"]["error_type"] == "segmentation"}
    treatment_seg = {item["id"] for item in per_case if item["treatment"]["error_type"] == "segmentation"}
    baseline_correct = {item["id"] for item in per_case if item["baseline"]["exact_match"]}
    treatment_correct = {item["id"] for item in per_case if item["treatment"]["exact_match"]}
    transitions = Counter((item["baseline"]["error_type"], item["treatment"]["error_type"]) for item in per_case)

    all_ids = {item["id"] for item in per_case}
    document = {
        "config": {
            "dictionary": str(args.dictionary),
            "cases": str(args.cases),
            "baseline_weights": str(args.baseline_weights),
            "treatment_weights": str(args.treatment_weights),
            "top_k": args.top_k,
            "beam_width": args.beam_width,
            "cascade_threshold": args.cascade_threshold,
        },
        "case_count": len(per_case),
        "per_case": per_case,
        "baseline_errors": baseline_errors,
        "treatment_errors": treatment_errors,
        "baseline_by_type": summarize(per_case, "baseline")["by_type"],
        "treatment_by_type": summarize(per_case, "treatment")["by_type"],
        "fixed": len((all_ids - baseline_correct) & treatment_correct),
        "regressed": len(baseline_correct & treatment_seg),
        "still_wrong": len((all_ids - baseline_correct) & (all_ids - treatment_correct)),
        "segmentation_table": {
            "both_segmentation": len(baseline_seg & treatment_seg),
            "resolved": len(baseline_seg - treatment_seg),
            "induced": len(treatment_seg - baseline_seg),
            "neither": len(all_ids - baseline_seg - treatment_seg),
        },
        "error_type_transitions": {f"{before}->{after}": count for (before, after), count in sorted(transitions.items())},
        "baseline_summary": summarize(per_case, "baseline"),
        "treatment_summary": summarize(per_case, "treatment"),
    }
    # "fixed" and "still_wrong" are correctness-paired statuses, while
    # segmentation_table is the primary segmentation comparison.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
