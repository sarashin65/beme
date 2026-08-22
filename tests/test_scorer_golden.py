"""Bit-exact regression test for the frozen legacy K-best baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from beme.dictionary import Dictionary
from beme.lattice import build_lattice, k_best_viterbi


ROOT = Path(__file__).resolve().parents[1]
DICTIONARY = ROOT / "data" / "dictionary.tsv"
GOLDEN = ROOT / "tests" / "data" / "kbest_golden_legacy_v1.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_kbest_golden_is_bit_exact() -> None:
    lines = GOLDEN.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0])["metadata"]
    assert metadata["dictionary_sha256"] == _sha256(DICTIONARY)
    assert metadata["k"] == 10

    dictionary = Dictionary.from_tsv(DICTIONARY)
    records = [json.loads(line) for line in lines[1:]]
    assert len(records) >= 80
    for record in records:
        paths = k_best_viterbi(build_lattice(record["reading"], dictionary, allow_literals=True), beam_width=10)
        expected = record["paths"]
        assert paths
        assert len(paths) == len(expected)
        for actual, wanted in zip(paths, expected):
            assert repr(actual.cost) == repr(float(wanted["cost"]))
            assert actual.surface == wanted["surface"]
            assert [
                [token.reading, token.surface, token.pos, token.cost] for token in actual.tokens
            ] == wanted["tokens"]
