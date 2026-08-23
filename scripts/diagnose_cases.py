#!/usr/bin/env python3
"""Produce deterministic before/after and stratified conversion diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beme.benchmark import katakana_to_hiragana, load_mozc_cases  # noqa: E402
from beme.converter import Converter  # noqa: E402
from beme.dictionary import Dictionary  # noqa: E402
from beme.evaluation import load_cases  # noqa: E402
from beme.features import extractor_for, path_features  # noqa: E402
from beme.lattice import LatticePath  # noqa: E402
from beme.oracle import constrained_best_path  # noqa: E402
from beme.scorer import LinearScorer  # noqa: E402
from beme.training_data import TrainingExample, load_jsonl  # noqa: E402
from beme.weights import Weights  # noqa: E402


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    reading: str
    gold: str
    example: TrainingExample | None = None


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _display_path(value: Path) -> str:
    try:
        return str(value.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return value.name if value.is_absolute() else str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="d1 JSONL split")
    source.add_argument("--suite", choices=("handmade", "mozc", "ajimee"))
    parser.add_argument("--dictionary", type=Path, default=ROOT / "data" / "dictionary_v2.tsv")
    parser.add_argument("--baseline-weights", type=Path, required=True)
    parser.add_argument("--learned-weights", type=Path, required=True)
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument("--beam-width", type=_positive_int, default=64)
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_cases(args: argparse.Namespace) -> tuple[list[Case], Path]:
    if args.data:
        examples = load_jsonl(args.data)
        return [Case(example.id or str(index), example.reading, example.target_surface, example)
                for index, example in enumerate(examples)], args.data
    if args.suite == "ajimee":
        raise ValueError("AJIMEE is sentence-level and is not supported by case diagnosis")
    if args.suite == "handmade":
        path = ROOT / "data" / "evaluation_cases.jsonl"
        cases = load_cases(path)
        return [Case(case["id"], case["reading"], max(case["judgments"], key=lambda item: (case["judgments"][item], item)))
                for case in cases], path
    path = ROOT / "data" / "mozc_oss.tsv"
    cases = load_mozc_cases(path)
    selected = [case for case in cases if case.command == "Conversion Match" or case.expected_rank is not None]
    return [Case(case.id, katakana_to_hiragana(case.reading), case.expected) for case in selected], path


def _top1(result: Any) -> str | None:
    return result.candidates[0].surface if result.candidates else None


def _rank(result: Any, gold: str) -> int | None:
    for index, candidate in enumerate(result.candidates, 1):
        if candidate.surface == gold:
            return index
    return None


def _candidate_list(result: Any, top_k: int) -> list[dict[str, Any]]:
    return [{"surface": candidate.surface, "cost": candidate.cost} for candidate in result.candidates[:top_k]]


def _contribution(
    baseline_path: LatticePath,
    learned_path: LatticePath,
    baseline_extractor: Any,
    learned_extractor: Any,
    baseline_weights: Weights,
    learned_weights: Weights,
) -> dict[str, Any]:
    baseline_features = path_features(baseline_path, baseline_extractor, baseline_weights)
    learned_features_for_baseline = path_features(learned_path, baseline_extractor, baseline_weights)
    baseline_features_for_learned = path_features(baseline_path, learned_extractor, learned_weights)
    learned_features = path_features(learned_path, learned_extractor, learned_weights)
    names = sorted(
        set(baseline_features)
        | set(learned_features_for_baseline)
        | set(baseline_features_for_learned)
        | set(learned_features)
    )
    rows = []
    for name in names:
        baseline_delta = learned_features_for_baseline.get(name, 0.0) - baseline_features.get(name, 0.0)
        learned_delta = learned_features.get(name, 0.0) - baseline_features_for_learned.get(name, 0.0)
        rows.append({
            "feature": name,
            "baseline_feature_delta": baseline_delta,
            "learned_feature_delta": learned_delta,
            "baseline_weight_delta": baseline_weights[name] * baseline_delta,
            "learned_weight_delta": learned_weights[name] * learned_delta,
        })
    rows.sort(key=lambda row: (-max(abs(row["baseline_weight_delta"]), abs(row["learned_weight_delta"])), row["feature"]))
    baseline_delta = sum(row["baseline_weight_delta"] for row in rows)
    learned_delta = sum(row["learned_weight_delta"] for row in rows)
    return {
        "features": rows,
        "baseline_weight_total": baseline_delta,
        "learned_weight_total": learned_delta,
        "baseline_path_cost_difference": baseline_delta,
        "learned_path_cost_difference": learned_delta,
    }


def _bucket(example: TrainingExample, axis: str, proper_noun: bool) -> str:
    if axis == "reading_length":
        length = len(example.reading)
        for label, low, high in (("1-3", 1, 3), ("4-6", 4, 6), ("7-10", 7, 10), ("11-16", 11, 16)):
            if low <= length <= high:
                return label
        return "out_of_range"
    if axis == "proper_noun":
        return "true" if proper_noun else "false"
    if axis == "numeric_reading":
        words = "いち に さん し ご ろく なな はち きゅう じゅう ひゃく せん まん".split()
        return "true" if any(word in example.reading for word in words) or re.search(r"[0-9]", example.target_surface) else "false"
    if axis == "span_tokens":
        return str(example.span_tokens) if example.span_tokens is not None else "unknown"
    if axis == "contrast_pair":
        return "true" if example.contrast_pair_id is not None else "false"
    raise ValueError(f"unknown stratification axis: {axis}")


def _stratify(cases: list[Case], records: list[dict[str, Any]], proper: dict[str, bool]) -> dict[str, Any]:
    axes = ("reading_length", "proper_noun", "numeric_reading", "span_tokens", "contrast_pair")
    result: dict[str, Any] = {}
    caveat = "span_tokens and contrast_pair are proxy metrics for MeCab conversion quality, not conversion quality itself."
    by_id = {case.id: case for case in cases}
    for axis in axes:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            example = by_id[record["id"]].example
            if example is None:
                continue
            groups[_bucket(example, axis, proper.get(record["id"], False))].append(record)
        axis_result = {}
        for label in sorted(groups):
            group = groups[label]
            axis_result[label] = {
                "examples": len(group),
                "low_support": len(group) < 30,
                "baseline": {"end_to_end_error_rate": sum(not item["baseline"]["correct"] for item in group) / len(group)},
                "learned": {"end_to_end_error_rate": sum(not item["learned"]["correct"] for item in group) / len(group)},
                "proxy_metric": axis in {"span_tokens", "contrast_pair"},
                "caveat": caveat if axis in {"span_tokens", "contrast_pair"} else None,
            }
        result[axis] = axis_result
    return result


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".diagnose-cases-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.suite == "ajimee":
        raise SystemExit("AJIMEE is sentence-level and is not supported by case diagnosis")
    if args.stratify and not args.data:
        raise SystemExit("--stratify requires --data")
    cases, dataset_path = _load_cases(args)
    dictionary = Dictionary.from_tsv(args.dictionary)
    baseline_weights = Weights.load(args.baseline_weights, feature_spec=None)
    learned_weights = Weights.load(args.learned_weights, feature_spec=None)
    baseline_extractor = extractor_for(baseline_weights.feature_spec, dictionary.reading_conditional)
    learned_extractor = extractor_for(learned_weights.feature_spec, dictionary.reading_conditional)
    baseline_scorer = LinearScorer(baseline_extractor, baseline_weights)
    learned_scorer = LinearScorer(learned_extractor, learned_weights)
    baseline_converter = Converter(dictionary, beam_width=args.beam_width, scorer=baseline_scorer)
    learned_converter = Converter(dictionary, beam_width=args.beam_width, scorer=learned_scorer)

    records: list[dict[str, Any]] = []
    proper: dict[str, bool] = {}
    for case in cases:
        lattice = baseline_converter.build_lattice(case.reading)
        baseline = baseline_converter.convert(case.reading, k=args.beam_width, allow_literals=True)
        learned = learned_converter.convert(case.reading, k=args.beam_width, allow_literals=True)
        baseline_top = _top1(baseline)
        learned_top = _top1(learned)
        baseline_correct = baseline_top == case.gold
        learned_correct = learned_top == case.gold
        if baseline_correct and learned_correct:
            status = "unchanged"
        elif not baseline_correct and not learned_correct:
            status = "still_wrong"
        elif learned_correct:
            status = "fixed"
        else:
            status = "regressed"
        record: dict[str, Any] = {
            "id": case.id,
            "reading": case.reading,
            "gold": case.gold,
            "baseline": {"top1": baseline_top, "gold_rank": _rank(baseline, case.gold), "top5": _candidate_list(baseline, args.top_k), "correct": baseline_correct},
            "learned": {"top1": learned_top, "gold_rank": _rank(learned, case.gold), "top5": _candidate_list(learned, args.top_k), "correct": learned_correct},
            "status": status,
        }
        if case.example is not None:
            gold_path = constrained_best_path(lattice, case.gold, baseline_scorer)
            proper[case.id] = bool(gold_path and any(token.pos == "proper_noun" for token in gold_path.tokens))
        if baseline_top != learned_top and baseline_top is not None and learned_top is not None:
            baseline_path = constrained_best_path(lattice, baseline_top, baseline_scorer)
            learned_path = constrained_best_path(lattice, learned_top, learned_scorer)
            if baseline_path is not None and learned_path is not None:
                record["contribution"] = _contribution(
                    baseline_path, learned_path, baseline_extractor, learned_extractor,
                    baseline_weights, learned_weights,
                )
        records.append(record)

    document: dict[str, Any] = {
        "dataset": {"path": _display_path(dataset_path), "examples": len(cases), "sha256": _sha256(dataset_path)},
        "dictionary_sha256": _sha256(args.dictionary),
        "baseline_weights": {"path": _display_path(args.baseline_weights), "feature_spec": baseline_weights.feature_spec, "sha256": _sha256(args.baseline_weights)},
        "learned_weights": {"path": _display_path(args.learned_weights), "feature_spec": learned_weights.feature_spec, "sha256": _sha256(args.learned_weights)},
        "top_k": args.top_k,
        "beam_width": args.beam_width,
        "cases": records,
        "representative": {
            "fixed": sorted((item for item in records if item["status"] == "fixed"), key=lambda item: item["id"])[:20],
            "regressed": sorted((item for item in records if item["status"] == "regressed"), key=lambda item: item["id"])[:20],
        },
    }
    if args.stratify:
        document["stratification"] = _stratify(cases, records, proper)
    _atomic_json(args.output, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
