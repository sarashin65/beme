#!/usr/bin/env python3
"""Run deterministic IR evaluation for the BeME converter."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beme.converter import Converter  # noqa: E402
from beme.evaluation import evaluate, load_cases  # noqa: E402
from beme.reranker import (  # noqa: E402
    DEFAULT_LLM_CONTEXT_SIZE,
    DEFAULT_LLM_MODEL_PATH,
    DEFAULT_LLM_WEIGHT,
    DEFAULT_VITERBI_WEIGHT,
    LLMLogitsReranker,
    LLMRerankerUnavailable,
    MockContextReranker,
    Reranker,
)


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
    parser.add_argument("--reranker", choices=("mock", "llm"), default="mock")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LLM_MODEL_PATH)
    parser.add_argument("--llm-weight", type=_threshold, default=DEFAULT_LLM_WEIGHT)
    parser.add_argument("--viterbi-weight", type=_threshold, default=DEFAULT_VITERBI_WEIGHT)
    parser.add_argument(
        "--llm-context-size",
        type=_positive_int,
        default=DEFAULT_LLM_CONTEXT_SIZE,
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    return parser


class _MeasuredReranker:
    def __init__(self, reranker: Reranker) -> None:
        self.reranker = reranker
        self.elapsed_seconds = 0.0
        self.candidate_count = 0

    def rerank(self, candidates: Any, context: Any) -> Any:
        started = perf_counter()
        result = self.reranker.rerank(candidates, context)
        self.elapsed_seconds += perf_counter() - started
        self.candidate_count += len(candidates)
        return result

    @property
    def candidate_latency_ms(self) -> float | None:
        if not self.candidate_count:
            return None
        return self.elapsed_seconds * 1000.0 / self.candidate_count


def _load_reranker(args: argparse.Namespace) -> tuple[Reranker, str]:
    if args.reranker == "mock":
        return MockContextReranker(), "mock"
    try:
        return (
            LLMLogitsReranker(
                args.model_path,
                llm_weight=args.llm_weight,
                viterbi_weight=args.viterbi_weight,
                context_size=args.llm_context_size,
            ),
            "llm",
        )
    except LLMRerankerUnavailable as exc:
        print(f"warning: {exc}; falling back to mock reranker", file=sys.stderr)
        return MockContextReranker(), "mock"


def _human_report(report: dict[str, object]) -> str:
    k = report["k"]
    return "\n".join(
        (
            f"Reranker: {report['reranker_effective']}"
            f" (requested: {report['reranker_requested']})",
            f"Queries: {report['query_count']}",
            f"NDCG@{k}: {float(report['ndcg_at_k']):.6f}",
            f"MRR@{k}: {float(report['mrr_at_k']):.6f}",
            f"Top-1 accuracy: {float(report['top1_accuracy']):.6f}",
            f"Rerank rate: {float(report['rerank_rate']):.6f}",
            f"Context rerank rate: {float(report['context_rerank_rate']):.6f}",
            f"Candidate latency: {float(report['candidate_latency_ms']):.3f} ms"
            if report["candidate_latency_ms"] is not None
            else "Candidate latency: n/a",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        reranker, effective_reranker = _load_reranker(args)
        measured_reranker = _MeasuredReranker(reranker)
        converter = Converter.from_tsv(
            args.dictionary,
            beam_width=args.beam_width,
            reranker=measured_reranker,
        )
        cases = load_cases(args.cases)
        report = evaluate(
            converter,
            cases,
            k=args.top_k,
            cascade_threshold=args.cascade_threshold,
        )
        report["reranker_requested"] = args.reranker
        report["reranker_effective"] = effective_reranker
        report["candidate_latency_ms"] = measured_reranker.candidate_latency_ms
        report["scored_candidate_count"] = measured_reranker.candidate_count
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(_human_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
