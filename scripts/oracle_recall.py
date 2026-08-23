#!/usr/bin/env python3
"""Measure lattice coverage, path recall, and surface recall separately."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beme.benchmark import katakana_to_hiragana, load_mozc_cases  # noqa: E402
from beme.converter import Converter  # noqa: E402
from beme.dictionary import Dictionary  # noqa: E402
from beme.evaluation import load_cases  # noqa: E402
from beme.features import extractor_for  # noqa: E402
from beme.lattice import k_best_viterbi  # noqa: E402
from beme.oracle import constrained_best_path  # noqa: E402
from beme.scorer import LinearScorer  # noqa: E402
from beme.training_data import load_jsonl  # noqa: E402
from beme.weights import Weights  # noqa: E402


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


def _parse_k(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(item) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("k must be comma-separated positive integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("k must be comma-separated positive integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="d1 JSONL split")
    source.add_argument("--suite", choices=("mozc", "handmade", "ajimee"))
    parser.add_argument("--dictionary", type=Path, default=ROOT / "data" / "dictionary_v2.tsv")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--k", type=_parse_k, default=(1, 5, 10, 20, 64))
    parser.add_argument("--beam-width", type=_positive_int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="also print JSON to stdout")
    return parser


def _suite_examples(suite: str) -> list[tuple[str, str]]:
    if suite == "handmade":
        return [
            (case["reading"], max(case["judgments"], key=lambda surface: (case["judgments"][surface], surface)))
            for case in load_cases(ROOT / "data" / "evaluation_cases.jsonl")
        ]
    if suite == "mozc":
        cases = load_mozc_cases(ROOT / "data" / "mozc_oss.tsv")
        # Negative regression assertions do not define a gold surface.
        return [
            (katakana_to_hiragana(case.reading), case.expected)
            for case in cases
            if case.command == "Conversion Match" or case.expected_rank is not None
        ]
    raise ValueError("AJIMEE is sentence-level and is not supported by oracle recall")


def _data_examples(path: Path) -> list[tuple[str, str]]:
    return [(example.reading, example.target_surface) for example in load_jsonl(path)]


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(percentile) - 1]) if len(values) > 1 else float(values[0])


def measure(
    examples: Iterable[tuple[str, str]],
    converter: Converter,
    scorer: LinearScorer,
    ks: tuple[int, ...],
    beam_width: int,
) -> dict[str, Any]:
    examples = list(examples)
    # Candidates are deduplicated and cost-sorted before the k cut, and only a
    # reranker could reorder them, so convert(k=max) is convert(k) for every
    # smaller k.  Converting once per example instead of once per k is what
    # keeps this affordable on the degenerate ablations.
    if converter.reranker is not None:
        raise ValueError("oracle recall requires a converter without a reranker")
    largest = max(ks)
    covered = 0
    path_hits = {k: 0 for k in ks}
    surface_hits = {k: 0 for k in ks}
    unique_counts: list[int] = []
    for reading, gold in examples:
        lattice = converter.build_lattice(reading)
        is_covered = constrained_best_path(lattice, gold, scorer) is not None
        covered += is_covered
        paths = k_best_viterbi(lattice, beam_width=largest, scorer=scorer)
        surfaces = {path.surface for path in paths[:beam_width]}
        unique_counts.append(len(surfaces))
        path_rank = next((rank for rank, path in enumerate(paths, 1) if path.surface == gold), None)
        candidates = converter.convert(reading, k=largest, allow_literals=True).candidates
        surface_rank = next(
            (rank for rank, candidate in enumerate(candidates, 1) if candidate.surface == gold), None
        )
        for k in ks:
            if path_rank is not None and path_rank <= k:
                path_hits[k] += 1
            if surface_rank is not None and surface_rank <= k:
                surface_hits[k] += 1
    total = len(examples)
    coverage_rate = covered / total if total else 0.0
    result = {
        "lattice_coverage": {"covered": covered, "examples": total, "rate": coverage_rate},
        "path_recall_at_k": {str(k): path_hits[k] / total if total else 0.0 for k in ks},
        "surface_recall_at_k": {str(k): surface_hits[k] / total if total else 0.0 for k in ks},
        "unique_surface_count": {
            "mean": statistics.mean(unique_counts) if unique_counts else 0.0,
            "p50": _percentile(unique_counts, 50),
            "p90": _percentile(unique_counts, 90),
            "max": max(unique_counts, default=0),
        },
    }
    path_rates = list(result["path_recall_at_k"].values())
    surface_rates = list(result["surface_recall_at_k"].values())
    if path_rates != sorted(path_rates) or surface_rates != sorted(surface_rates):
        raise ValueError("recall must be monotonic in k")
    if any(rate > coverage_rate + 1e-12 for rate in path_rates):
        raise ValueError("path recall cannot exceed lattice coverage")
    path_at_beam = result["path_recall_at_k"].get(str(beam_width))
    if path_at_beam is not None and any(rate > path_at_beam + 1e-12 for rate in surface_rates):
        raise ValueError("surface recall cannot exceed path recall at beam width")
    if ks[0] == 1:
        # Both are "the gold surface is the top-ranked one", reached by two
        # independent routes (k_best_viterbi here, Converter.convert there),
        # so a disagreement means the two are not scoring the same model.
        if not math.isclose(path_rates[0], surface_rates[0], abs_tol=1e-12):
            raise ValueError("path recall@1 and surface recall@1 disagree")
    return result


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".oracle-recall-", dir=path.parent)
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
        raise SystemExit("AJIMEE is sentence-level and is not supported by oracle recall")
    if max(args.k) > args.beam_width:
        raise SystemExit("--k values must not exceed --beam-width")
    dictionary = Dictionary.from_tsv(args.dictionary)
    weights = Weights.load(args.weights, feature_spec=None)
    extractor = extractor_for(weights.feature_spec, dictionary.reading_conditional)
    scorer = LinearScorer(extractor, weights)
    converter = Converter(dictionary, beam_width=args.beam_width, scorer=scorer)
    examples = _data_examples(args.data) if args.data else _suite_examples(args.suite)
    measured = measure(examples, converter, scorer, args.k, args.beam_width)
    source_path = args.data or (ROOT / "data" / ("evaluation_cases.jsonl" if args.suite == "handmade" else "mozc_oss.tsv"))
    document = {
        "dataset": {"path": _display_path(source_path), "examples": len(examples), "sha256": _sha256(source_path)},
        "dictionary_sha256": _sha256(args.dictionary),
        "weights": {"path": _display_path(args.weights), "feature_spec": weights.feature_spec, "sha256": _sha256(args.weights)},
        "beam_width": args.beam_width,
        **measured,
    }
    _atomic_json(args.output, document)
    if args.json:
        print(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
