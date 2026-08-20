"""Focused tests for dictionary lookup and lattice connectivity."""

from __future__ import annotations

from pathlib import Path

import pytest

from beme.dictionary import Dictionary
from beme.lattice import build_lattice


def write_dictionary(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "dictionary.tsv"
    path.write_text("reading\tsurface\tpos\tcost\n" + text, encoding="utf-8")
    return path


def test_dictionary_lookup_returns_all_entries_at_a_position(tmp_path: Path) -> None:
    path = write_dictionary(
        tmp_path,
        "かな\t仮名\tnoun\t1\nかな\t佳名\tproper\t2\nか\t可\tnoun\t0\n",
    )
    dictionary = Dictionary.from_tsv(path)

    entries = dictionary.matches_at("かな", 0)

    assert {(entry.reading, entry.surface) for entry in entries} == {
        ("か", "可"),
        ("かな", "仮名"),
        ("かな", "佳名"),
    }
    assert dictionary.matches_at("かな", 1) == ()


def test_lattice_contains_multi_segment_edges_and_unknown_fallback(tmp_path: Path) -> None:
    dictionary = Dictionary.from_tsv(
        write_dictionary(tmp_path, "かな\t仮名\tnoun\t1\nか\t可\tnoun\t0\n")
    )

    lattice = build_lattice("かなみ", dictionary)

    assert [node.position for node in lattice.nodes] == [0, 1, 2, 3]
    assert any(edge.start == 0 and edge.end == 2 and edge.token.surface == "仮名" for edge in lattice.nodes[0].edges)
    assert any(edge.start == 2 and edge.end == 3 and edge.token.reading == "み" for edge in lattice.nodes[2].edges)
    assert all(node.edges for node in lattice.nodes[:-1])


def test_lattice_rejects_invalid_reading(tmp_path: Path) -> None:
    dictionary = Dictionary.from_tsv(write_dictionary(tmp_path, "かな\t仮名\tnoun\t1\n"))

    with pytest.raises(ValueError):
        build_lattice("カナ", dictionary)
