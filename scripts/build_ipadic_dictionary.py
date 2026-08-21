#!/usr/bin/env python3
"""Build BeME's TSV dictionary from fixed MeCab IPADIC source CSV files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beme.ipadic import (  # noqa: E402
    DEFAULT_BEME_COST_CEILING,
    DEFAULT_BEME_COST_FLOOR,
    DEFAULT_RAW_COST_CEILING,
    DEFAULT_RAW_COST_FLOOR,
    build_dictionary,
    sha256_file,
)

DEFAULT_OVERLAY = REPOSITORY_ROOT / "data" / "dictionary_manual.tsv"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "dictionary.tsv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MeCab IPADIC source CSV files to BeME's dictionary TSV"
    )
    parser.add_argument("source", type=Path, help="IPADIC source directory or .tar.gz archive")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overlay",
        type=Path,
        action="append",
        default=None,
        help="BeME TSV whose entries override imported entries (repeatable; default: data/dictionary_manual.tsv)",
    )
    parser.add_argument("--encoding", default="euc_jp", help="source CSV encoding (default: euc_jp)")
    parser.add_argument("--raw-cost-floor", type=int, default=DEFAULT_RAW_COST_FLOOR)
    parser.add_argument("--raw-cost-ceiling", type=int, default=DEFAULT_RAW_COST_CEILING)
    parser.add_argument("--beme-cost-floor", type=float, default=DEFAULT_BEME_COST_FLOOR)
    parser.add_argument("--beme-cost-ceiling", type=float, default=DEFAULT_BEME_COST_CEILING)
    parser.add_argument(
        "--expect-sha256",
        help="reject an archive whose SHA-256 differs (recommended for reproducible builds)",
    )
    parser.add_argument("--json", action="store_true", help="print build statistics as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expect_sha256:
        if not args.source.is_file():
            raise SystemExit("--expect-sha256 requires an archive file source")
        actual_digest = sha256_file(args.source)
        if actual_digest.lower() != args.expect_sha256.lower():
            raise SystemExit(
                f"source SHA-256 mismatch: expected {args.expect_sha256.lower()}, got {actual_digest}"
            )
    overlays = args.overlay if args.overlay is not None else [DEFAULT_OVERLAY]
    stats = build_dictionary(
        args.source,
        args.output,
        overlays=overlays,
        encoding=args.encoding,
        raw_floor=args.raw_cost_floor,
        raw_ceiling=args.raw_cost_ceiling,
        beme_floor=args.beme_cost_floor,
        beme_ceiling=args.beme_cost_ceiling,
    )
    payload = {
        "source": str(args.source),
        "output": str(args.output),
        "source_rows": stats.source_rows,
        "imported_entries": stats.imported_entries,
        "duplicate_rows": stats.duplicate_rows,
        "skipped_symbols": stats.skipped_symbols,
        "skipped_readings": stats.skipped_readings,
        "overlay_entries": stats.overlay_entries,
        "output_entries": stats.output_entries,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "Built {output_entries} entries from {source_rows} IPADIC rows "
            "({duplicate_rows} duplicates, {skipped_symbols} symbols, "
            "{skipped_readings} invalid readings skipped; {overlay_entries} overlay entries) "
            "at {output}".format(**payload)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
