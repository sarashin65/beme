#!/usr/bin/env python3
"""Run deterministic IR evaluation for the BeME converter."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beme.converter import Converter  # noqa: E402
from beme.evaluation import evaluate, load_cases  # noqa: E402


DEFAULT_DICTIONARY = REPOSITORY_ROOT / "data" / "dictionary.tsv"
DEFAULT_CASES = REPOSITORY_ROOT / "data" / "evaluation_cases.jsonl"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _threshold(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate BeME conversion ranking")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument("--beam-width", type=_positive_int, default=64)
    parser.add_argument("--cascade-threshold", type=_threshold, default=2.0)
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    return parser


def _human_report(report: dict[str, object]) -> str:
    k = report["k"]
    return "\n".join(
        (
            f"Queries: {report['query_count']}",
            f"NDCG@{k}: {float(report['ndcg_at_k']):.6f}",
            f"MRR@{k}: {float(report['mrr_at_k']):.6f}",
            f"Top-1 accuracy: {float(report['top1_accuracy']):.6f}",
            f"Rerank rate: {float(report['rerank_rate']):.6f}",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        converter = Converter.from_tsv(args.dictionary, beam_width=args.beam_width)
        cases = load_cases(args.cases)
        report = evaluate(
            converter,
            cases,
            k=args.top_k,
            cascade_threshold=args.cascade_threshold,
        )
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(_human_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
