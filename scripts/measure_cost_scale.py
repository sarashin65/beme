#!/usr/bin/env python3
"""Measure confidence distributions and cost-scale recalibration inputs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beme.benchmark import katakana_to_hiragana, load_ajimee_cases, load_mozc_cases  # noqa: E402
from beme.converter import Converter  # noqa: E402
from beme.dictionary import Dictionary  # noqa: E402
from beme.evaluation import load_cases  # noqa: E402
from beme.features import extractor_for  # noqa: E402
from beme.training_data import load_jsonl  # noqa: E402
from beme.scorer import LinearScorer  # noqa: E402
from beme.weights import Weights  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name if path.is_absolute() else str(path)


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def _threshold(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="d1 JSONL split")
    source.add_argument("--suite", choices=("handmade", "mozc", "ajimee"))
    parser.add_argument("--dictionary", type=Path, default=ROOT / "data" / "dictionary_v2.tsv")
    parser.add_argument("--baseline-weights", type=Path, required=True)
    parser.add_argument("--learned-weights", type=Path, required=True)
    parser.add_argument("--baseline-threshold", type=_threshold, default=2.0)
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument("--beam-width", type=_positive_int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def _quantile(values: list[float], fraction: float) -> float:
    """Return the linearly interpolated inclusive quantile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def confidence_summary(values: Iterable[float | None], baseline_threshold: float) -> dict[str, Any]:
    all_values = list(values)
    finite = [float(value) for value in all_values if value is not None]
    ordered = sorted(finite)
    baseline_rate = sum(value < baseline_threshold for value in finite) / len(finite) if finite else 0.0
    threshold_index = min(int(round(baseline_rate * len(ordered))), max(len(ordered) - 1, 0))
    recalibrated = ordered[threshold_index] if ordered else None
    return {
        "count": len(finite),
        "null_confidence_count": len(all_values) - len(finite),
        "p10": _quantile(finite, 0.10),
        "p25": _quantile(finite, 0.25),
        "p50": _quantile(finite, 0.50),
        "p75": _quantile(finite, 0.75),
        "p90": _quantile(finite, 0.90),
        "mean": statistics.mean(finite) if finite else 0.0,
        "min": min(finite, default=0.0),
        "max": max(finite, default=0.0),
        "baseline_fire_rate": baseline_rate,
        "recalibrated_threshold": recalibrated,
        "recalibrated_fire_rate": sum(value < recalibrated for value in finite) / len(finite) if finite and recalibrated is not None else 0.0,
    }


def _confidence_values(readings: Iterable[str], converter: Converter, top_k: int, allow_literals: bool = True) -> list[float | None]:
    values = []
    for reading in readings:
        values.append(converter.convert(reading, k=top_k, allow_literals=allow_literals).confidence)
    return values


def _suite_readings(suite: str, path: Path) -> list[str]:
    if suite == "handmade":
        return [case["reading"] for case in load_cases(path)]
    cases = load_mozc_cases(path)
    return [katakana_to_hiragana(case.reading) for case in cases if case.supported]


def _ajimee_readings(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    readings = []
    for item in payload:
        splits = item["splitted_input_for_limited_input_length"]
        readings.extend(splits or [item["input"]])
    return [katakana_to_hiragana(reading) for reading in readings]


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".cost-scale-", dir=path.parent)
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
    dictionary = Dictionary.from_tsv(args.dictionary)
    baseline = Weights.load(args.baseline_weights, feature_spec=None)
    learned = Weights.load(args.learned_weights, feature_spec=None)
    baseline_converter = Converter(dictionary, beam_width=args.beam_width, scorer=LinearScorer(extractor_for(baseline.feature_spec, dictionary.reading_conditional), baseline))
    learned_converter = Converter(dictionary, beam_width=args.beam_width, scorer=LinearScorer(extractor_for(learned.feature_spec, dictionary.reading_conditional), learned))
    sections: dict[str, Any] = {}
    if args.data:
        readings = [example.reading for example in load_jsonl(args.data)]
        sections["d1"] = (readings, args.data)
    elif args.suite == "ajimee":
        path = ROOT / "data" / "ajimee_bench_jwtd_v2_v1.json"
        sections["ajimee"] = (_ajimee_readings(path), path)
    else:
        path = ROOT / "data" / ("evaluation_cases.jsonl" if args.suite == "handmade" else "mozc_oss.tsv")
        sections[args.suite] = (_suite_readings(args.suite, path), path)
    distributions = {}
    for name, (readings, _) in sections.items():
        base_values = _confidence_values(readings, baseline_converter, args.top_k)
        learned_values = _confidence_values(readings, learned_converter, args.top_k)
        base_summary = confidence_summary(base_values, args.baseline_threshold)
        learned_summary = confidence_summary(learned_values, args.baseline_threshold)
        if base_values == learned_values:
            learned_summary["recalibrated_threshold"] = args.baseline_threshold
            learned_summary["recalibrated_fire_rate"] = base_summary["baseline_fire_rate"]
        baseline_p = base_summary["p50"]
        learned_p = learned_summary["p50"]
        distributions[name] = {
            "examples": len(readings),
            "baseline": base_summary,
            "learned": learned_summary,
            "scale_ratio": learned_p / baseline_p if baseline_p else None,
            "p25_ratio": learned_summary["p25"] / base_summary["p25"] if base_summary["p25"] else None,
            "p75_ratio": learned_summary["p75"] / base_summary["p75"] if base_summary["p75"] else None,
            "viterbi_weight_initial": 1.0 / (learned_p / baseline_p) if learned_p and baseline_p else None,
            "rate_baseline": base_summary["baseline_fire_rate"],
            "threshold_learned": learned_summary["recalibrated_threshold"],
        }
    findings = {
        "needs_rerank_is_unused": {
            "confirmed": True,
            "detail": "Verified by source search: converter.py computes needs_rerank at line 174, while reranked is gated at line 176; only evaluation.py reads needs_rerank for aggregation.",
            "references": ["src/beme/converter.py:174", "src/beme/converter.py:176", "src/beme/evaluation.py:250"],
            "impact": "cascade_threshold does not affect accuracy metrics; recalibration is a statistic, not an optimization.",
        },
        "viterbi_weight_out_of_scope": {
            "reason": "viterbi_weight is used by CrossEncoderReranker and LLMLogitsReranker only; wiring K-best into a reranker is deferred to Phase 4.",
        },
    }
    document = {
        "datasets": {name: {"path": _display_path(path), "examples": len(readings), "sha256": _sha256(path)} for name, (readings, path) in sections.items()},
        "baseline_weights": {"path": _display_path(args.baseline_weights), "feature_spec": baseline.feature_spec, "sha256": _sha256(args.baseline_weights)},
        "learned_weights": {"path": _display_path(args.learned_weights), "feature_spec": learned.feature_spec, "sha256": _sha256(args.learned_weights)},
        "baseline_threshold": args.baseline_threshold,
        "top_k": args.top_k,
        "beam_width": args.beam_width,
        "quantile_method": "linear interpolation (inclusive quantile for p10/p25/p50/p75/p90); recalibration threshold uses the empirical upper boundary at the baseline fire count",
        "confidence_distributions": distributions,
        "design_findings": findings,
    }
    _atomic_json(args.output, document)
    if args.json:
        print(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
