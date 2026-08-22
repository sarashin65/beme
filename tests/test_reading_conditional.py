"""Reading-conditional lexical feature tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from beme.dictionary import Dictionary, DictionaryEntry
from beme.features import extractor_for
from beme.lattice import LatticeEdge, Token


def _dictionary() -> Dictionary:
    return Dictionary(
        (
            DictionaryEntry("ゆいいつ", "唯一", "noun", 2.0, raw_cost=5000),
            DictionaryEntry("こうしょう", "交渉", "noun", 2.0, raw_cost=3000),
            DictionaryEntry("こうしょう", "高尚", "adjective", 3.0, raw_cost=7000),
            DictionaryEntry("こうしょう", "こうしょう", "literal", 0.0),
        )
    )


def _features(dictionary: Dictionary, reading: str, surface: str, pos: str, raw_cost: int | None) -> dict[str, float]:
    extractor = extractor_for("decomposed_v1+freq_v1", dictionary.reading_conditional)
    return extractor.lexical_features(
        LatticeEdge(0, len(reading), Token(reading, surface, pos, 1.0, raw_cost=raw_cost))
    )


def test_unique_reading_has_zero_logprob_and_log_two_count() -> None:
    features = _features(_dictionary(), "ゆいいつ", "唯一", "noun", 5000)
    assert features["lex.reading_cond_logprob"] == pytest.approx(0.0)
    assert features["lex.reading_candidate_count"] == pytest.approx(math.log(2.0))


def test_ambiguous_reading_preserves_raw_cost_order() -> None:
    dictionary = _dictionary()
    lower = _features(dictionary, "こうしょう", "交渉", "noun", 3000)
    higher = _features(dictionary, "こうしょう", "高尚", "adjective", 7000)
    assert lower["lex.reading_cond_logprob"] < higher["lex.reading_cond_logprob"]
    assert lower["lex.ipadic_cost_mapped"] != 0.0


def test_conditional_probabilities_sum_to_one() -> None:
    dictionary = _dictionary()
    values = [
        _features(dictionary, "こうしょう", "交渉", "noun", 3000),
        _features(dictionary, "こうしょう", "高尚", "adjective", 7000),
    ]
    assert sum(math.exp(-item["lex.reading_cond_logprob"]) for item in values) == pytest.approx(1.0)


def test_rawless_entries_have_count_but_no_logprob() -> None:
    features = _features(_dictionary(), "こうしょう", "こうしょう", "literal", None)
    assert "lex.reading_cond_logprob" not in features
    assert features["lex.reading_candidate_count"] == pytest.approx(math.log(4.0))


def test_extreme_raw_costs_are_finite() -> None:
    dictionary = Dictionary(
        (
            DictionaryEntry("かな", "A", "noun", 1.0, raw_cost=-100000),
            DictionaryEntry("かな", "B", "noun", 1.0, raw_cost=100000),
        )
    )
    for surface, raw_cost in (("A", -100000), ("B", 100000)):
        features = _features(dictionary, "かな", surface, "noun", raw_cost)
        assert math.isfinite(features["lex.reading_cond_logprob"])


def test_frequency_spec_requires_a_dictionary_table() -> None:
    extractor = extractor_for("decomposed_v1+freq_v1")
    edge = LatticeEdge(0, 2, Token("かな", "仮名", "noun", 1.0, raw_cost=1000))
    with pytest.raises(ValueError, match="reading conditional table"):
        extractor.lexical_features(edge)


def test_base_specs_do_not_gain_frequency_features() -> None:
    dictionary = _dictionary()
    edge = LatticeEdge(0, 4, Token("ゆいいつ", "唯一", "noun", 2.0, raw_cost=5000))
    assert "lex.reading_cond_logprob" not in extractor_for("decomposed_v1", dictionary.reading_conditional).lexical_features(edge)
