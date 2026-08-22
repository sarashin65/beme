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
from beme.features import legacy_v1_extractor  # noqa: E402
from beme.scorer import LinearScorer  # noqa: E402
from beme.weights import DEFAULT_WEIGHTS_PATH, Weights  # noqa: E402
from beme.benchmark import (  # noqa: E402
    evaluate_ajimee,
    evaluate_mozc,
    load_ajimee_cases,
    load_mozc_cases,
)
from beme.evaluation import evaluate, load_cases  # noqa: E402
from beme.reranker import (  # noqa: E402
    DEFAULT_CROSS_ENCODER_BATCH_SIZE,
    DEFAULT_CROSS_ENCODER_CACHE_DIR,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_WEIGHT,
    DEFAULT_LLM_CONTEXT_SIZE,
    DEFAULT_LLM_MODEL_PATH,
    DEFAULT_LLM_WEIGHT,
    DEFAULT_VITERBI_WEIGHT,
    SUPPORTED_CROSS_ENCODER_MODELS,
    CrossEncoderReranker,
    CrossEncoderRerankerUnavailable,
    LLMLogitsReranker,
    LLMRerankerUnavailable,
    MockContextReranker,
    Reranker,
)


DEFAULT_DICTIONARY = REPOSITORY_ROOT / "data" / "dictionary.tsv"
DEFAULT_CASES = REPOSITORY_ROOT / "data" / "evaluation_cases.jsonl"
DEFAULT_AJIMEE_CASES = REPOSITORY_ROOT / "data" / "ajimee_bench_jwtd_v2_v1.json"
DEFAULT_MOZC_CASES = REPOSITORY_ROOT / "data" / "mozc_oss.tsv"


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
    parser = argparse.ArgumentParser(description="Evaluate BeME conversion quality")
    parser.add_argument(
        "--suite",
        choices=("handmade", "ajimee", "mozc", "all"),
        default="handmade",
    )
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--ajimee-cases", type=Path, default=DEFAULT_AJIMEE_CASES)
    parser.add_argument("--mozc-cases", type=Path, default=DEFAULT_MOZC_CASES)
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument(
        "--rerank-k",
        dest="rerank_k",
        type=_positive_int,
        default=None,
        help="number of candidates fed to the reranker (default: every candidate)",
    )
    parser.add_argument("--beam-width", type=_positive_int, default=64)
    parser.add_argument("--cascade-threshold", type=_threshold, default=2.0)
    parser.add_argument(
        "--reranker",
        choices=("mock", "llm", "cross-encoder"),
        default="mock",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LLM_MODEL_PATH)
    parser.add_argument("--llm-weight", type=_threshold, default=DEFAULT_LLM_WEIGHT)
    parser.add_argument("--viterbi-weight", type=_threshold, default=DEFAULT_VITERBI_WEIGHT)
    parser.add_argument(
        "--llm-context-size",
        type=_positive_int,
        default=DEFAULT_LLM_CONTEXT_SIZE,
    )
    parser.add_argument(
        "--cross-encoder-model",
        choices=SUPPORTED_CROSS_ENCODER_MODELS,
        default=DEFAULT_CROSS_ENCODER_MODEL,
    )
    parser.add_argument(
        "--cross-encoder-cache-dir",
        type=Path,
        default=DEFAULT_CROSS_ENCODER_CACHE_DIR,
    )
    parser.add_argument(
        "--cross-encoder-weight",
        type=_threshold,
        default=DEFAULT_CROSS_ENCODER_WEIGHT,
    )
    parser.add_argument(
        "--cross-encoder-batch-size",
        type=_positive_int,
        default=DEFAULT_CROSS_ENCODER_BATCH_SIZE,
    )
    parser.add_argument("--cross-encoder-local-files-only", action="store_true")
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
    if args.reranker == "llm":
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
    try:
        return (
            CrossEncoderReranker(
                args.cross_encoder_model,
                cross_encoder_weight=args.cross_encoder_weight,
                viterbi_weight=args.viterbi_weight,
                batch_size=args.cross_encoder_batch_size,
                cache_dir=args.cross_encoder_cache_dir,
                local_files_only=args.cross_encoder_local_files_only,
            ),
            "cross-encoder",
        )
    except CrossEncoderRerankerUnavailable as exc:
        print(f"warning: {exc}; falling back to mock reranker", file=sys.stderr)
        return MockContextReranker(), "mock"


def _human_report(report: dict[str, object]) -> str:
    suite = report["suite"]
    model = (
        f", model: {report['cross_encoder_model']}"
        if report["reranker_effective"] == "cross-encoder"
        else ""
    )
    reranker = (
        f"Reranker: {report['reranker_effective']}"
        f" (requested: {report['reranker_requested']}{model})"
    )
    latency = (
        f"Candidate latency: {float(report['candidate_latency_ms']):.3f} ms"
        if report["candidate_latency_ms"] is not None
        else "Candidate latency: n/a"
    )
    widths = (
        f"Widths: top-k={report['top_k']}, rerank-k="
        f"{'all' if report['rerank_k'] is None else report['rerank_k']}"
    )
    if suite == "handmade":
        k = report["k"]
        lines = (
            f"Queries: {report['query_count']}",
            f"NDCG@{k}: {float(report['ndcg_at_k']):.6f}",
            f"MRR@{k}: {float(report['mrr_at_k']):.6f}",
            f"Top-1 accuracy: {float(report['top1_accuracy']):.6f}",
            f"Rerank rate: {float(report['rerank_rate']):.6f}",
            f"Context rerank rate: {float(report['context_rerank_rate']):.6f}",
        )
    elif suite == "ajimee":
        lines = (
            f"Cases: {report['case_count']}",
            f"Mean CER: {float(report['mean_cer']):.6f}",
            f"Mean edit distance: {float(report['mean_edit_distance']):.6f}",
            f"Exact-match accuracy: {float(report['exact_match_accuracy']):.6f}",
        )
    elif suite == "mozc":
        lines = (
            f"Cases: {report['case_count']}",
            f"Evaluated: {report['evaluated_count']}",
            f"Passed: {report['passed_count']}",
            f"Pass rate: {float(report['pass_rate']):.6f}",
            f"Skipped: {report['skipped_count']}",
        )
    else:
        handmade = report["handmade"]
        ajimee = report["ajimee"]
        mozc = report["mozc"]
        lines = (
            f"Handmade: {handmade['query_count']} cases, "
            f"NDCG@{handmade['k']}={float(handmade['ndcg_at_k']):.6f}, "
            f"MRR@{handmade['k']}={float(handmade['mrr_at_k']):.6f}, "
            f"Top-1={float(handmade['top1_accuracy']):.6f}",
            f"AJIMEE: {ajimee['case_count']} cases, "
            f"mean CER={float(ajimee['mean_cer']):.6f}, "
            f"exact={float(ajimee['exact_match_accuracy']):.6f}",
            f"Mozc: {mozc['passed_count']}/{mozc['evaluated_count']} passed "
            f"({float(mozc['pass_rate']):.6f}), {mozc['skipped_count']} skipped",
        )
    return "\n".join((reranker, *lines, widths, latency))


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
            rerank_k=args.rerank_k,
            scorer=LinearScorer(legacy_v1_extractor(), Weights.load(args.weights)),
        )
        reports: dict[str, dict[str, object]] = {}
        if args.suite in {"handmade", "all"}:
            reports["handmade"] = evaluate(
                converter,
                load_cases(args.cases),
                k=args.top_k,
                cascade_threshold=args.cascade_threshold,
            )
            reports["handmade"]["suite"] = "handmade"
        if args.suite in {"ajimee", "all"}:
            reports["ajimee"] = evaluate_ajimee(
                converter,
                load_ajimee_cases(args.ajimee_cases),
                k=args.top_k,
                cascade_threshold=args.cascade_threshold,
            )
        if args.suite in {"mozc", "all"}:
            reports["mozc"] = evaluate_mozc(
                converter,
                load_mozc_cases(args.mozc_cases),
                k=args.top_k,
                cascade_threshold=args.cascade_threshold,
            )
        if args.suite == "all":
            report: dict[str, object] = {"suite": "all", **reports}
        else:
            report = reports[args.suite]
        report["reranker_requested"] = args.reranker
        report["reranker_effective"] = effective_reranker
        report["cross_encoder_model"] = (
            args.cross_encoder_model if effective_reranker == "cross-encoder" else None
        )
        report["viterbi_weight"] = args.viterbi_weight
        report["top_k"] = args.top_k
        report["rerank_k"] = args.rerank_k
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
