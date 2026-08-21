#!/usr/bin/env python3
"""Stream a Japanese Wikipedia subset into a plain-text corpus file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "corpus.txt"
DEFAULT_DATASET = "wikimedia/wikipedia"
DEFAULT_CONFIG = "20231101.ja"
DEFAULT_TARGET_BYTES = 100 * 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Japanese Wikipedia articles as a plain-text corpus"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output path (default: data/corpus.txt)",
    )
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=DEFAULT_TARGET_BYTES,
        help="stop after approximately this many UTF-8 bytes (default: 104857600)",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        help="stop after this many articles; useful for a smoke test",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--revision",
        help="optional Hugging Face dataset revision for reproducible downloads",
    )
    parser.add_argument("--json", action="store_true", help="print statistics as JSON")
    return parser


def fetch_corpus(args: argparse.Namespace) -> tuple[int, int]:
    if args.target_bytes <= 0:
        raise ValueError("--target-bytes must be greater than zero")
    if args.max_articles is not None and args.max_articles <= 0:
        raise ValueError("--max-articles must be greater than zero")

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "The 'datasets' package is required; run 'uv sync --group dev' first."
        ) from error

    load_options = {"split": args.split, "streaming": True}
    if args.revision:
        load_options["revision"] = args.revision
    articles = load_dataset(args.dataset, args.config, **load_options)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    article_count = 0
    byte_count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for article in articles:
            text = article.get("text")
            if not isinstance(text, str) or not text:
                continue

            # Keep the file valid UTF-8 even when the target falls in a multibyte
            # character. Articles are separated so the corpus remains line based.
            separator = "\n" if byte_count else ""
            available = args.target_bytes - byte_count - len(separator.encode("utf-8"))
            if available <= 0:
                break
            encoded = text.encode("utf-8")[:available]
            chunk = encoded.decode("utf-8", errors="ignore")
            if not chunk:
                break
            output.write(separator)
            output.write(chunk)
            byte_count += len(separator.encode("utf-8")) + len(chunk.encode("utf-8"))
            article_count += 1
            if byte_count >= args.target_bytes:
                break
            if args.max_articles is not None and article_count >= args.max_articles:
                break

    return article_count, byte_count


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        article_count, byte_count = fetch_corpus(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = {"articles": article_count, "bytes": byte_count, "output": str(args.output)}
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"Wrote {byte_count:,} bytes from {article_count:,} articles to {args.output}")
    return 0


if __name__ == "__main__":
    import os

    status = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
