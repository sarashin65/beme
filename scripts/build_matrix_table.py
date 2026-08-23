#!/usr/bin/env python3
"""Extract the dictionary-relevant portion of IPADIC matrix.def."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beme.dictionary import Dictionary  # noqa: E402
from beme.matrix_cost import MAGIC, VERSION  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_def", type=Path)
    parser.add_argument("--dictionary", type=Path, default=ROOT / "data" / "dictionary_v2.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "ipadic_matrix_v1.bin")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ids = {0}
    dictionary = Dictionary.from_tsv(args.dictionary)
    for entry in dictionary.entries:
        if entry.left_id >= 0:
            ids.add(entry.left_id)
        if entry.right_id >= 0:
            ids.add(entry.right_id)
    ordered_ids = tuple(sorted(ids))
    wanted = set(ordered_ids)
    values: dict[tuple[int, int], int] = {}
    costs: list[int] = []
    with args.matrix_def.open(encoding="ascii") as stream:
        dimensions = stream.readline().split()
        if len(dimensions) != 2:
            raise ValueError("matrix.def must start with row and column dimensions")
        for line_number, line in enumerate(stream, 2):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"matrix.def line {line_number}: expected three integers")
            left, right, cost = (int(field) for field in fields)
            if left in wanted and right in wanted:
                values[(left, right)] = cost
                costs.append(cost)
    expected = len(ordered_ids) ** 2
    if len(values) != expected:
        raise ValueError(f"matrix.def contains {len(values)} selected pairs, expected {expected}")
    payload = bytearray(struct.pack("<8sHHI", MAGIC, VERSION, 0, len(ordered_ids)))
    payload.extend(struct.pack(f"<{len(ordered_ids)}i", *ordered_ids))
    payload.extend(struct.pack(
        f"<{expected}h",
        *(values[(row, column)] for row in ordered_ids for column in ordered_ids),
    ))
    args.output.write_bytes(payload)
    digest = hashlib.sha256(args.matrix_def.read_bytes()).hexdigest()
    report = {
        "id_count": len(ordered_ids),
        "pair_count": expected,
        "cost_min": min(costs),
        "cost_max": max(costs),
        "cost_mean": sum(costs) / len(costs),
        "input_sha256": digest,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True) if args.json else args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
