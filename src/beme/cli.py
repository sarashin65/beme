"""Command-line interface for BeME kana conversion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Sequence

from .converter import Converter, ConversionResult
from .reranker import (
    DEFAULT_CROSS_ENCODER_BATCH_SIZE,
    DEFAULT_CROSS_ENCODER_CACHE_DIR,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_WEIGHT,
    DEFAULT_LLM_CONTEXT_SIZE,
    DEFAULT_LLM_MODEL_PATH,
    DEFAULT_LLM_WEIGHT,
    DEFAULT_VITERBI_WEIGHT,
    SUPPORTED_CROSS_ENCODER_MODELS,
    ConversionContext,
    CrossEncoderReranker,
    CrossEncoderRerankerUnavailable,
    LLMLogitsReranker,
    LLMRerankerUnavailable,
    MockContextReranker,
    Reranker,
)


_DEFAULT_DICTIONARY = Path(__file__).resolve().parents[2] / "data" / "dictionary.tsv"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beme", description="Deterministic hiragana-to-kanji conversion mock")
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert", help="convert one hiragana reading")
    convert.add_argument("reading", help="non-empty hiragana reading (ー allowed)")
    convert.add_argument("--top-k", type=_positive_int, default=5, help="number of candidates (default: 5)")
    convert.add_argument("--beam-width", type=_positive_int, default=64, help="paths retained per node (default: 64)")
    convert.add_argument(
        "--cascade-threshold",
        type=float,
        default=2.0,
        help="rerank when the top-two cost gap is below this value (default: 2.0)",
    )
    convert.add_argument(
        "--dictionary",
        type=Path,
        default=_DEFAULT_DICTIONARY,
        help=f"TSV dictionary (default: {_DEFAULT_DICTIONARY})",
    )
    convert.add_argument(
        "--preceding-context",
        default="",
        help="confirmed text before the current reading",
    )
    convert.add_argument(
        "--following-context",
        default="",
        help="text after the cursor",
    )
    convert.add_argument(
        "--reranker",
        choices=("mock", "llm", "cross-encoder"),
        default="mock",
        help="context reranker backend (default: mock)",
    )
    convert.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_LLM_MODEL_PATH,
        help=f"GGUF model for --reranker llm (default: {DEFAULT_LLM_MODEL_PATH})",
    )
    convert.add_argument(
        "--llm-weight",
        type=_non_negative_float,
        default=DEFAULT_LLM_WEIGHT,
        help=f"mean negative log-likelihood weight (default: {DEFAULT_LLM_WEIGHT:g})",
    )
    convert.add_argument(
        "--viterbi-weight",
        type=_non_negative_float,
        default=DEFAULT_VITERBI_WEIGHT,
        help=f"Viterbi cost weight (default: {DEFAULT_VITERBI_WEIGHT:g})",
    )
    convert.add_argument(
        "--llm-context-size",
        type=_positive_int,
        default=DEFAULT_LLM_CONTEXT_SIZE,
        help=f"llama.cpp context token limit (default: {DEFAULT_LLM_CONTEXT_SIZE})",
    )
    convert.add_argument(
        "--cross-encoder-model",
        choices=SUPPORTED_CROSS_ENCODER_MODELS,
        default=DEFAULT_CROSS_ENCODER_MODEL,
        help=f"model for --reranker cross-encoder (default: {DEFAULT_CROSS_ENCODER_MODEL})",
    )
    convert.add_argument(
        "--cross-encoder-cache-dir",
        type=Path,
        default=DEFAULT_CROSS_ENCODER_CACHE_DIR,
        help=f"repo-external model cache (default: {DEFAULT_CROSS_ENCODER_CACHE_DIR})",
    )
    convert.add_argument(
        "--cross-encoder-weight",
        type=_non_negative_float,
        default=DEFAULT_CROSS_ENCODER_WEIGHT,
        help=f"semantic relevance weight (default: {DEFAULT_CROSS_ENCODER_WEIGHT:g})",
    )
    convert.add_argument(
        "--cross-encoder-batch-size",
        type=_positive_int,
        default=DEFAULT_CROSS_ENCODER_BATCH_SIZE,
        help=f"CPU inference batch size (default: {DEFAULT_CROSS_ENCODER_BATCH_SIZE})",
    )
    convert.add_argument(
        "--cross-encoder-local-files-only",
        action="store_true",
        help="do not download a missing Cross-Encoder model",
    )
    convert.add_argument("--json", action="store_true", help="emit one JSON object")
    return parser


def _build_reranker(args: argparse.Namespace) -> Reranker:
    if args.reranker == "mock":
        return MockContextReranker()
    if args.reranker == "llm":
        try:
            return LLMLogitsReranker(
                args.model_path,
                llm_weight=args.llm_weight,
                viterbi_weight=args.viterbi_weight,
                context_size=args.llm_context_size,
            )
        except LLMRerankerUnavailable as exc:
            print(f"warning: {exc}; falling back to mock reranker", file=sys.stderr)
            return MockContextReranker()
    try:
        return CrossEncoderReranker(
            args.cross_encoder_model,
            cross_encoder_weight=args.cross_encoder_weight,
            viterbi_weight=args.viterbi_weight,
            batch_size=args.cross_encoder_batch_size,
            cache_dir=args.cross_encoder_cache_dir,
            local_files_only=args.cross_encoder_local_files_only,
        )
    except CrossEncoderRerankerUnavailable as exc:
        print(f"warning: {exc}; falling back to mock reranker", file=sys.stderr)
        return MockContextReranker()


def _print_text(result: ConversionResult) -> None:
    print(f"reading: {result.reading}")
    for index, candidate in enumerate(result.candidates, start=1):
        score = f"\tscore={candidate.score:.6f}" if result.reranked else ""
        print(f"{index}. {candidate.surface}\tcost={candidate.cost:.6f}{score}")
    confidence = "None" if result.confidence is None else f"{result.confidence:.6f}"
    print(f"confidence: {confidence}")
    print(f"cascade_threshold: {result.cascade_threshold:g}")
    print(f"needs_rerank: {'yes' if result.needs_rerank else 'no'}")
    print(f"preceding_context: {json.dumps(result.context.preceding, ensure_ascii=False)}")
    print(f"following_context: {json.dumps(result.context.following, ensure_ascii=False)}")
    print(f"reranked: {'yes' if result.reranked else 'no'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "convert":
        parser.error("a command is required")
    try:
        converter = Converter.from_tsv(
            args.dictionary,
            beam_width=args.beam_width,
            reranker=_build_reranker(args),
        )
        result = converter.convert(
            args.reading,
            k=args.top_k,
            cascade_threshold=args.cascade_threshold,
            context=ConversionContext(
                preceding=args.preceding_context,
                following=args.following_context,
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
    else:
        _print_text(result)
    return 0


__all__ = ["build_parser", "main"]
