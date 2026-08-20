"""Behavioral tests for the Phase 2 reranker contract and mock."""

from __future__ import annotations

from pathlib import Path

import pytest

from beme.converter import Converter
from beme.reranker import ConversionContext, MockContextReranker


def write_dictionary(tmp_path: Path) -> Path:
    path = tmp_path / "dictionary.tsv"
    path.write_text(
        "reading\tsurface\tpos\tcost\n"
        "こうしょう\t交渉\tnoun\t0.0\n"
        "こうしょう\t考証\tnoun\t0.4\n"
        "こうしょう\t校章\tnoun\t0.5\n",
        encoding="utf-8",
    )
    return path


def test_preceding_context_promotes_history_sense_from_full_candidate_pool(tmp_path: Path) -> None:
    converter = Converter.from_tsv(write_dictionary(tmp_path), reranker=MockContextReranker())

    baseline = converter.convert("こうしょう", k=1)
    contextual = converter.convert(
        "こうしょう",
        k=1,
        context=ConversionContext(preceding="歴史資料を考える"),
    )

    assert baseline.candidates[0].surface == "交渉"
    assert baseline.reranked is False
    assert contextual.candidates[0].surface == "考証"
    assert contextual.candidates[0].cost == pytest.approx(0.4)
    assert contextual.candidates[0].score == pytest.approx(-0.6)
    assert contextual.reranked is True


def test_following_context_promotes_business_sense_deterministically(tmp_path: Path) -> None:
    converter = Converter.from_tsv(write_dictionary(tmp_path), reranker=MockContextReranker())
    context = ConversionContext(following="について商談を進める")

    first = converter.convert("こうしょう", k=3, context=context)
    second = converter.convert("こうしょう", k=3, context=context)

    assert first == second
    assert first.candidates[0].surface == "交渉"
    assert first.candidates[0].score == pytest.approx(-1.0)
    assert {candidate.surface for candidate in first.candidates} == {"交渉", "考証", "校章"}


def test_mock_reranker_supports_injected_keyword_lexicon(tmp_path: Path) -> None:
    reranker = MockContextReranker(
        keyword_bonus=2.0,
        keywords={"校章": ("制服",)},
    )
    converter = Converter.from_tsv(write_dictionary(tmp_path), reranker=reranker)

    result = converter.convert(
        "こうしょう",
        k=1,
        context=ConversionContext(preceding="制服を確認する"),
    )

    assert result.candidates[0].surface == "校章"
    assert result.candidates[0].score == pytest.approx(-1.5)


def test_context_rejects_non_string_fields() -> None:
    with pytest.raises(TypeError, match="preceding context"):
        ConversionContext(preceding=1)  # type: ignore[arg-type]
