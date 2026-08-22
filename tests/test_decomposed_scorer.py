"""Tolerance-based equivalence checks for the decomposed Phase 1 scorer."""

from __future__ import annotations

import json
import math
from pathlib import Path

from beme.dictionary import Dictionary
from beme.features import decomposed_v1_extractor, legacy_v1_extractor, path_features
from beme.lattice import build_lattice, k_best_viterbi
from beme.scorer import LinearScorer
from beme.weights import Weights


ROOT = Path(__file__).resolve().parents[1]


def test_decomposed_v1_matches_legacy_golden_surfaces_and_costs() -> None:
    dictionary = Dictionary.from_tsv(ROOT / "data/dictionary_v2.tsv")
    legacy = LinearScorer(legacy_v1_extractor(), Weights.load(ROOT / "data/weights/legacy_v1.json"))
    decomposed = LinearScorer(
        decomposed_v1_extractor(),
        Weights.load(ROOT / "data/weights/decomposed_v1.json", feature_spec="decomposed_v1"),
    )
    records = [
        json.loads(line)
        for line in (ROOT / "tests/data/kbest_golden_legacy_v1.jsonl").read_text().splitlines()[1:]
    ]
    for record in records:
        lattice = build_lattice(record["reading"], dictionary, allow_literals=True)
        expected = k_best_viterbi(lattice, beam_width=10, scorer=legacy)
        actual = k_best_viterbi(lattice, beam_width=10, scorer=decomposed)
        assert [path.surface for path in actual] == [path.surface for path in expected]
        for expected_path, actual_path in zip(expected, actual, strict=True):
            assert math.isclose(actual_path.cost, expected_path.cost, abs_tol=1e-9)
            features = path_features(actual_path, decomposed.extractor)
            reconstructed = sum(decomposed.weights[name] * value for name, value in features.items())
            assert math.isclose(reconstructed, actual_path.cost, abs_tol=1e-9)
