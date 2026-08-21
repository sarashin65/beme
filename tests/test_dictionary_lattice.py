"""Focused tests for dictionary lookup and lattice connectivity."""

from __future__ import annotations

from pathlib import Path

import pytest

from beme.dictionary import Dictionary
from beme.lattice import backward_viterbi, build_lattice, k_best_viterbi, pos_transition_cost


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


def test_backward_costs_include_token_and_transition_costs(tmp_path: Path) -> None:
    dictionary = Dictionary.from_tsv(
        write_dictionary(
            tmp_path,
            "かな\t仮名\tnoun\t1\n"
            "か\t可\tnoun\t0\n"
            "な\tな\tparticle\t2\n",
        )
    )
    lattice = build_lattice("かな", dictionary)

    costs = backward_viterbi(lattice)

    assert costs[-1] == 0.0
    assert costs[1] == 2.0
    assert costs[0] == 1.0
    assert costs[0] == 1.0 + costs[2]


def test_backward_cost_matches_forward_best_path(tmp_path: Path) -> None:
    dictionary = Dictionary.from_tsv(
        write_dictionary(tmp_path, "か\t可\tnoun\t1\nか\t蚊\tparticle\t0.5\n")
    )
    lattice = build_lattice("かか", dictionary)

    backward = backward_viterbi(lattice)
    forward = lattice.paths_at(1, beam_width=8)
    expected = min(
        path.cost + edge.token.cost + pos_transition_cost(
            path.token.pos if path.token else None, edge.token.pos
        )
        for path in forward
        for edge in lattice.nodes[1].edges
    )

    assert backward[1] == min(edge.token.cost for edge in lattice.nodes[1].edges)
    assert backward[0] == expected


def test_k_best_is_exactly_ordered_without_beam_pruning(tmp_path: Path) -> None:
    dictionary = Dictionary.from_tsv(
        write_dictionary(
            tmp_path,
            "かな\t仮名\tnoun\t1.0\n"
            "か\t佳\tnoun\t0.0\n"
            "か\t蚊\tparticle\t0.4\n"
            "な\t名\tnoun\t0.0\n"
            "な\t菜\tparticle\t0.3\n",
        )
    )
    lattice = build_lattice("かな", dictionary)

    paths = k_best_viterbi(lattice, beam_width=5)

    assert [path.cost for path in paths] == sorted(path.cost for path in paths)
    assert [path.surface for path in paths] == ["佳名", "佳菜", "蚊名", "蚊菜", "仮名"]
    assert [path.cost for path in paths] == pytest.approx([0.08, 0.35, 0.43, 0.78, 1.0])


def test_exact_k_best_keeps_locally_expensive_global_best(tmp_path: Path) -> None:
    """A narrow local beam must not discard a path that wins later."""

    dictionary = Dictionary.from_tsv(
        write_dictionary(
            tmp_path,
            "か\t低\tnoun\t0.0\n"
            "か\t高\tparticle\t0.05\n"
            "き\tく\tverb\t0.0\n",
        )
    )
    lattice = build_lattice("かき", dictionary)

    exact = k_best_viterbi(lattice, beam_width=1)
    locally_pruned = lattice.paths_at(2, beam_width=1)

    assert exact[0].surface == "高く"
    assert exact[0].cost == pytest.approx(0.09)
    assert locally_pruned[0].surface == "低く"
    assert all(path.surface != "高く" for path in locally_pruned)
