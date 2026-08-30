"""Corpus-frequency lexical feature tests (issue #19 F2)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import beme.features as features_module
from beme.corpus_freq import CORPUS_LOGPROB_TEMPERATURE, CorpusFrequencyTable
from beme.features import (
    CORPUS_LEXICAL_FEATURES,
    FeatureExtractor,
    path_features,
    resolve_spec,
)
from beme.lattice import LatticeEdge, LatticePath, Token
from beme.weights import Weights


ROOT = Path(__file__).resolve().parents[1]
SPEC_ID = "decomposed_v1+corpus_v1"


def _table() -> CorpusFrequencyTable:
    # Hand-built table satisfying the builder's invariants (from_pairs ==
    # surface_counts, sum(pairs) == total) with all six counts distinct:
    # c(こう,高)=3, c(たか,高)=2, c(こう,甲)=1, c(こう)=4, c(高)=5, N=6.
    # Every numerator/denominator mix-up changes an expected value
    # (issue #19 task4 review M2 / re-review Minor 1).
    return CorpusFrequencyTable.from_counts(
        pairs={("こう", "高"): 3, ("たか", "高"): 2, ("こう", "甲"): 1},
        surfaces={"高": 5, "甲": 1},
        total=6,
    )


def _extractor(table: CorpusFrequencyTable | None) -> FeatureExtractor:
    return FeatureExtractor(resolve_spec(SPEC_ID), corpus_frequency=table)


def _edge(reading: str, surface: str, start: int = 0) -> LatticeEdge:
    return LatticeEdge(start, start + len(reading), Token(reading, surface, "noun", 1.0))


def test_observed_pair_matches_hand_computed_values() -> None:
    # c(r,s)=3, c(r)=4, c(s)=5, N=6, T=10 (hand-computed per mission 2.1):
    #   cond    = -log(3/4)/10
    #   unigram = -log(5/6)/10
    #   unseen  = 0.0
    features = _extractor(_table()).lexical_features(_edge("こう", "高"))
    assert features["lex.corpus_reading_cond_logprob"] == pytest.approx(-math.log(3.0 / 4.0) / CORPUS_LOGPROB_TEMPERATURE)
    assert features["lex.corpus_unigram_logprob"] == pytest.approx(-math.log(5.0 / 6.0) / CORPUS_LOGPROB_TEMPERATURE)
    assert features["lex.corpus_unseen"] == 0.0


def test_unobserved_pair_sets_unseen_flag_and_zero_conditional() -> None:
    # c(たか,甲)=0 but c(甲)=1 > 0: unseen flag on, conditional zero,
    # unigram valued.
    features = _extractor(_table()).lexical_features(_edge("たか", "甲"))
    assert features["lex.corpus_unseen"] == 1.0
    assert features["lex.corpus_reading_cond_logprob"] == 0.0
    assert features["lex.corpus_unigram_logprob"] == pytest.approx(-math.log(1.0 / 6.0) / CORPUS_LOGPROB_TEMPERATURE)


def test_unobserved_pair_and_unobserved_surface_zero_both_logprobs() -> None:
    # c(r,s)=0 and c(s)=0: both logprob features zero, unseen flag on.
    features = _extractor(_table()).lexical_features(_edge("しんじつ", "真実"))
    assert features["lex.corpus_unseen"] == 1.0
    assert features["lex.corpus_reading_cond_logprob"] == 0.0
    assert features["lex.corpus_unigram_logprob"] == 0.0


@pytest.mark.parametrize(
    ("base_spec_id", "corpus_spec_id"),
    (
        ("decomposed_v1", "decomposed_v1+corpus_v1"),
        (
            "decomposed_v1-no_script-surface_len",
            "decomposed_v1-no_script-surface_len+corpus_v1",
        ),
    ),
)
def test_specs_without_corpus_features_do_not_emit_them(base_spec_id: str, corpus_spec_id: str) -> None:
    # A0 (decomposed_v1) and A8 (the C1/C0 base conditions) output is
    # bit-identical to the pre-F2 feature set.
    table = _table()
    edge = _edge("こう", "高")
    base = FeatureExtractor(resolve_spec(base_spec_id)).lexical_features(edge)
    with_corpus = FeatureExtractor(resolve_spec(corpus_spec_id), corpus_frequency=table).lexical_features(edge)
    assert set(base) == set(with_corpus) - set(CORPUS_LEXICAL_FEATURES)
    for name in CORPUS_LEXICAL_FEATURES:
        assert name not in base
    for name, value in base.items():
        assert with_corpus[name] == value


def test_corpus_spec_without_table_raises_when_default_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        features_module, "_corpus_table",
        lambda: (_ for _ in ()).throw(FileNotFoundError(f"corpus frequency table not found: {tmp_path}")),
    )
    extractor = FeatureExtractor(resolve_spec(SPEC_ID))
    with pytest.raises(ValueError, match="corpus frequency table is required"):
        extractor.lexical_features(_edge("こう", "高"))


def test_new_spec_weight_files_load_strictly() -> None:
    for spec_id in ("decomposed_v1+corpus_v1", "decomposed_v1-no_script-surface_len+corpus_v1"):
        weights = Weights.load(ROOT / "data" / "weights" / "ablations" / f"{spec_id}.json", feature_spec=spec_id)
        for name in CORPUS_LEXICAL_FEATURES:
            assert weights[name] == 0.0


def test_corpus_features_are_position_independent() -> None:
    table = _table()
    extractor = _extractor(table)
    token = Token("こう", "高", "noun", 1.0)
    first = extractor.lexical_features(LatticeEdge(0, 2, token))
    second = extractor.lexical_features(LatticeEdge(7, 9, token))
    for name in CORPUS_LEXICAL_FEATURES:
        assert first[name] == second[name]


def test_path_features_sum_corpus_features_over_tokens() -> None:
    table = _table()
    extractor = _extractor(table)
    tokens = (
        Token("こう", "高", "noun", 1.0),
        Token("こう", "こう", "literal", 0.0),
    )
    path = LatticePath(0.0, tokens[1], LatticePath(0.0, tokens[0], None))
    features = path_features(path, extractor)
    per_token = [
        extractor.lexical_features(LatticeEdge(0, 2, tokens[0])),
        extractor.lexical_features(LatticeEdge(2, 4, tokens[1])),
    ]
    for name in CORPUS_LEXICAL_FEATURES:
        assert features[name] == pytest.approx(sum(item[name] for item in per_token))
    assert features["lex.corpus_unseen"] == 1.0
    assert features["lex.corpus_reading_cond_logprob"] == pytest.approx(
        -math.log(3.0 / 4.0) / CORPUS_LOGPROB_TEMPERATURE
    )
