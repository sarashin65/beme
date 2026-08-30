#!/usr/bin/env python3
"""Scale check for corpus features: std of the 3 corpus features on d1 test gold pairs.

Mission task4 §5-4 requires: std of each corpus feature <= 3x std of lex.ipadic_cost_mapped.
Reference std = 2.2423 (task1 measurement on d1).
Unlike measure_local_ceiling.py (which only aggregates DECOMPOSED_V1_SPEC features),
this walks the actual FeatureExtractor so the formulas under test are the real ones.
"""

from __future__ import annotations

from pathlib import Path
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beme.corpus_freq import CorpusFrequencyTable  # noqa: E402
from beme.features import CORPUS_LEXICAL_FEATURES, FeatureExtractor, resolve_spec  # noqa: E402
from beme.lattice import LatticeEdge, Token  # noqa: E402

SPEC_ID = "decomposed_v1+corpus_v1"
REF_STD = 2.2423  # lex.ipadic_cost_mapped on d1 (task1)


def report(values: list[float], name: str) -> float:
    mean = sum(values) / len(values)
    std = statistics.pstdev(values)
    print(f"{name:35s} mean={mean:.4f} std={std:.4f} min={min(values):.4f} max={max(values):.4f}")
    return std


def main() -> int:
    table = CorpusFrequencyTable.load(ROOT / "data" / "corpus_freq_v1.bin")
    extractor = FeatureExtractor(resolve_spec(SPEC_ID), corpus_frequency=table)

    values = {name: [] for name in CORPUS_LEXICAL_FEATURES}
    with open(ROOT / "data" / "d1" / "test.jsonl") as f:
        for line in f:
            d = json.loads(line)
            r, s = d["reading"], d["positive"]
            edge = LatticeEdge(0, len(r), Token(r, s, "noun", 1.0))
            feats = extractor.lexical_features(edge)
            for name in CORPUS_LEXICAL_FEATURES:
                values[name].append(feats[name])

    stds = {name: report(values[name], name) for name in CORPUS_LEXICAL_FEATURES}

    limit = REF_STD * 3
    worst = max(stds.values())
    print()
    print(f"基準: lex.ipadic_cost_mapped std={REF_STD} -> 3倍以内の上限 = {limit:.4f}")
    print(f"判定: max std = {worst:.4f} -> {'OK (3倍以内) / T=10.0 を学習前に固定' if worst <= limit else 'NG (T を再調整)'}")
    return 0 if worst <= limit else 1


if __name__ == "__main__":
    raise SystemExit(main())
