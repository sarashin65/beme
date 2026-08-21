"""Behavioral tests for the Phase 1 dictionary converter."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from beme.converter import Candidate, Converter
from beme.reranker import ConversionContext


def write_dictionary(tmp_path: Path, *rows: tuple[str, str, str, float]) -> Path:
    path = tmp_path / "dictionary.tsv"
    path.write_text(
        "reading\tsurface\tpos\tcost\n"
        + "".join(f"{reading}\t{surface}\t{pos}\t{cost}\n" for reading, surface, pos, cost in rows),
        encoding="utf-8",
    )
    return path


def test_lattice_keeps_direct_and_multi_token_segmentations(tmp_path: Path) -> None:
    dictionary = write_dictionary(
        tmp_path,
        ("かな", "仮名", "noun", 1.0),
        ("か", "佳", "noun", 0.2),
        ("な", "名", "noun", 0.2),
    )

    result = Converter.from_tsv(dictionary).convert("かな", k=5)
    surfaces = {candidate.surface for candidate in result.candidates}

    assert "仮名" in surfaces
    assert "佳名" in surfaces
    assert all("".join(token.reading for token in candidate.tokens) == "かな" for candidate in result.candidates)


def test_lattice_uses_one_character_unknown_fallback(tmp_path: Path) -> None:
    dictionary = write_dictionary(tmp_path, ("か", "可", "noun", 0.0))

    result = Converter.from_tsv(dictionary).convert("かみ", k=5)

    assert result.candidates
    assert all("".join(token.reading for token in candidate.tokens) == "かみ" for candidate in result.candidates)
    assert any(token.reading == "み" for candidate in result.candidates for token in candidate.tokens)


def test_unknown_only_input_still_has_a_connected_path(tmp_path: Path) -> None:
    dictionary = write_dictionary(tmp_path, ("あ", "阿", "noun", 0.0))

    result = Converter.from_tsv(dictionary).convert("ぬ", k=5)

    assert len(result.candidates) == 1
    assert result.candidates[0].tokens[0].reading == "ぬ"
    assert result.candidates[0].surface


def test_homophones_are_returned_once_and_kbest_order_is_deterministic(tmp_path: Path) -> None:
    dictionary = write_dictionary(
        tmp_path,
        ("かみ", "神", "noun", 0.0),
        ("かみ", "神", "proper", 1.0),
        ("かみ", "紙", "noun", 0.5),
        ("か", "佳", "noun", 0.1),
        ("み", "身", "noun", 0.1),
    )
    converter = Converter.from_tsv(dictionary, beam_width=16)

    first = converter.convert("かみ", k=4)
    second = converter.convert("かみ", k=4)

    assert first == second
    assert len(first.candidates) == len({candidate.surface for candidate in first.candidates})
    assert list(first.candidates) == sorted(first.candidates, key=lambda candidate: (candidate.cost, candidate.surface))
    assert {candidate.surface for candidate in first.candidates} >= {"神", "紙"}


def test_rerank_k_limits_reranker_input_without_changing_default(tmp_path: Path) -> None:
    dictionary = write_dictionary(
        tmp_path,
        ("かみ", "神", "noun", 0.0),
        ("かみ", "紙", "noun", 0.5),
        ("か", "佳", "noun", 0.1),
        ("み", "身", "noun", 0.1),
    )

    class RecordingReranker:
        def __init__(self) -> None:
            self.counts: list[int] = []

        def rerank(self, candidates, context):
            self.counts.append(len(candidates))
            return tuple(candidates)

    limited = RecordingReranker()
    result_limited = Converter.from_tsv(dictionary, reranker=limited, rerank_k=2).convert(
        "かみ", k=4, context=ConversionContext(preceding="文脈")
    )
    assert limited.counts == [2]
    assert len(result_limited.candidates) == 3

    unchanged = RecordingReranker()
    result = Converter.from_tsv(dictionary, reranker=unchanged).convert(
        "かみ", k=4, context=ConversionContext(preceding="文脈")
    )
    assert unchanged.counts == [len(result.candidates)]
    assert unchanged.counts[0] > 2


def test_confidence_boundary_is_strictly_below_threshold(tmp_path: Path) -> None:
    dictionary = write_dictionary(
        tmp_path,
        ("かみ", "神", "noun", 0.0),
        ("かみ", "紙", "noun", 2.0),
    )
    converter = Converter.from_tsv(dictionary)

    at_boundary = converter.convert("かみ", k=2, cascade_threshold=2.0)
    below_boundary = converter.convert("かみ", k=2, cascade_threshold=2.0001)

    assert at_boundary.confidence == pytest.approx(2.0)
    assert at_boundary.needs_rerank is False
    assert below_boundary.confidence == pytest.approx(2.0)
    assert below_boundary.needs_rerank is True


def test_single_candidate_has_no_confidence_or_rerank_request(tmp_path: Path) -> None:
    dictionary = write_dictionary(tmp_path, ("ねこ", "猫", "noun", 0.0))

    result = Converter.from_tsv(dictionary).convert("ねこ", k=1, cascade_threshold=100.0)

    assert result.confidence is None
    assert result.needs_rerank is False


def test_candidate_is_immutable() -> None:
    candidate = Candidate(surface="猫", cost=1.0, tokens=())

    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        candidate.surface = "犬"  # type: ignore[misc]


@pytest.mark.parametrize("reading", ["", "カナ", "かな漢", "かな!", "  "])
def test_invalid_reading_is_rejected(tmp_path: Path, reading: str) -> None:
    dictionary = write_dictionary(tmp_path, ("かな", "仮名", "noun", 0.0))

    with pytest.raises(ValueError):
        Converter.from_tsv(dictionary).convert(reading)


def test_long_vowel_mark_is_valid_input(tmp_path: Path) -> None:
    dictionary = write_dictionary(tmp_path, ("ー", "ー", "symbol", 0.0))

    result = Converter.from_tsv(dictionary).convert("ー", k=1)

    assert result.reading == "ー"
    assert result.candidates


@pytest.mark.parametrize(
    "row",
    [
        "かな\t仮名\tnoun",  # missing cost
        "カナ\t仮名\tnoun\t0",  # reading is not hiragana
        "\t仮名\tnoun\t0",  # empty reading
        "かな\t仮名\tnoun\tnot-a-number",  # malformed cost
    ],
)
def test_invalid_dictionary_data_is_rejected(tmp_path: Path, row: str) -> None:
    path = tmp_path / "invalid.tsv"
    path.write_text("reading\tsurface\tpos\tcost\n" + row + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        Converter.from_tsv(path)
