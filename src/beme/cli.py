"""Command-line interface for deterministic Phase 1 conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .converter import Converter, ConversionResult
from .reranker import ConversionContext, MockContextReranker


_DEFAULT_DICTIONARY = Path(__file__).resolve().parents[2] / "data" / "dictionary.tsv"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
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
    convert.add_argument("--json", action="store_true", help="emit one JSON object")
    return parser


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
            reranker=MockContextReranker(),
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
