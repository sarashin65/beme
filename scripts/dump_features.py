#!/usr/bin/env python3
"""Dump K-best path feature vectors as deterministic JSON Lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beme.dictionary import Dictionary
from beme.features import legacy_v1_extractor, path_features
from beme.lattice import build_lattice, k_best_viterbi
from beme.scorer import LinearScorer
from beme.weights import DEFAULT_WEIGHTS_PATH, Weights


def dump_features(
    dictionary_path: Path,
    weights_path: Path,
    k: int,
    cases_path: Path,
    output_path: Path,
) -> None:
    dictionary = Dictionary.from_tsv(dictionary_path)
    extractor = legacy_v1_extractor()
    scorer = LinearScorer(extractor, Weights.load(weights_path, feature_spec=extractor.spec.id))
    lines: list[str] = []
    for raw_reading in cases_path.read_text(encoding="utf-8").splitlines():
        reading = raw_reading.strip()
        if not reading or reading.startswith("#"):
            continue
        lattice = build_lattice(reading, dictionary, allow_literals=True)
        for rank, path in enumerate(k_best_viterbi(lattice, beam_width=k, scorer=scorer)):
            record = {
                "reading": reading,
                "surface": path.surface,
                "cost": path.cost,
                "rank": rank,
                "features": path_features(path, extractor, scorer.weights),
            }
            lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", type=Path, default=REPOSITORY_ROOT / "data" / "dictionary.tsv")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.k <= 0:
        raise SystemExit("--k must be positive")
    dump_features(args.dictionary, args.weights, args.k, args.cases, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
