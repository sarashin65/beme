"""Top-1 selection follows Converter's cost/surface tie rule."""

from __future__ import annotations

import random
import time

import pytest

from beme.converter import Converter
from beme.dictionary import Dictionary, DictionaryEntry
from beme.lattice import (
    Lattice,
    LatticeEdge,
    LatticeNode,
    Token,
    _top1_path,
    build_lattice,
    k_best_viterbi,
    top1_surface,
)
from beme.scorer import LinearScorer
from beme.training import evaluate_split
from beme.features import FeatureExtractor, LEGACY_V1_SPEC
from beme.training_data import TrainingExample
from beme.weights import Weights


def scorer() -> LinearScorer:
    return LinearScorer(
        FeatureExtractor(LEGACY_V1_SPEC),
        Weights({"lex.base_cost": 1.0, "trans.bos": 0.0}, feature_spec="legacy_v1"),
    )


def lattice(surfaces: list[str]) -> Lattice:
    return Lattice(
        "あ",
        (
            LatticeNode(
                0,
                tuple(LatticeEdge(0, 1, Token("あ", surface, "noun", 1.0)) for surface in surfaces),
            ),
            LatticeNode(1, ()),
        ),
    )


def converter(surfaces: list[str]) -> Converter:
    return Converter(
        Dictionary(DictionaryEntry("あ", surface, "noun", 1.0) for surface in surfaces),
        beam_width=64,
        scorer=scorer(),
    )


def test_top1_matches_converter_on_equal_cost_paths() -> None:
    surfaces = ["ん", "あ", "う"]
    assert top1_surface(lattice(surfaces), scorer()) == "あ"
    assert converter(surfaces).convert("あ", k=1).candidates[0].surface == "あ"


def test_top1_matches_first_kbest_path_without_ties() -> None:
    graph = Lattice(
        "あ",
        (
            LatticeNode(0, (LatticeEdge(0, 1, Token("あ", "あ", "noun", 1.0)),)),
            LatticeNode(1, ()),
        ),
    )
    model = scorer()
    assert top1_surface(graph, model) == k_best_viterbi(graph, beam_width=1, scorer=model)[0].surface


def test_top1_returns_none_for_unconnected_lattice() -> None:
    graph = Lattice("あ", (LatticeNode(0, ()), LatticeNode(1, ())))
    assert top1_surface(graph, scorer()) is None


def test_top1_expands_probe_for_more_than_eight_ties() -> None:
    surfaces = [chr(ord("ぁ") + index) for index in range(9)]
    assert top1_surface(lattice(surfaces), scorer(), tie_probe=8) == min(surfaces)


def test_evaluate_split_uses_converter_top1_for_ties() -> None:
    surfaces = ["ん", "あ", "う"]
    model = scorer()
    builder = lambda _reading: lattice(surfaces)
    evaluation = evaluate_split([TrainingExample("あ", "あ")], builder, model)
    assert evaluation.exact_matches == 1
    assert evaluation.model_errors == 0


def test_tie_probe_argument_is_accepted_but_no_longer_bounds_the_search() -> None:
    """Regression for issue #9 Phase 4 review item 2.

    The old implementation truncated tie enumeration at ``TOP1_MAX_TIE_PROBE``
    and silently returned the minimum of that truncated subset, breaking the
    documented "minimum (cost, surface) among all paths" contract. The
    frontier search has no bound to truncate, so an out-of-range tie_probe
    (larger than the old cap) must still find the true minimum.
    """

    surfaces = [chr(ord("ぁ") + index) for index in range(9)]
    assert top1_surface(lattice(surfaces), scorer(), tie_probe=1) == min(surfaces)
    assert top1_surface(lattice(surfaces), scorer(), tie_probe=100_000) == min(surfaces)


def _brute_force_minimum(graph: Lattice, model: LinearScorer) -> tuple[float, str] | None:
    """Enumerate every terminal path and return its true (cost, surface) minimum.

    Only used by tests on lattices small enough that full enumeration is
    feasible; production code must not do this (that combinatorial blow-up
    is exactly issue #9 Phase 4 review item 1).
    """

    terminal = len(graph.nodes) - 1
    best: tuple[float, str] | None = None

    def walk(position: int, previous_state, cost: float, surface: str) -> None:
        nonlocal best
        if position == terminal:
            candidate = (cost, surface)
            if best is None or candidate < best:
                best = candidate
            return
        for edge in graph.nodes[position].edges:
            next_cost = cost + model.edge_cost(edge) + model.transition_cost(previous_state, edge)
            walk(edge.end, model.state_key(edge), next_cost, surface + edge.token.surface)

    walk(0, None, 0.0, "")
    return best


def _random_tie_heavy_dictionary(rng: random.Random, reading: str) -> Dictionary:
    """Build a dictionary with many zero-cost, overlapping-span entries.

    Zero lexical cost for every entry mirrors the A2 (``decomposed_v1-trans_only``)
    ablation weight vector that motivated issue #9 Phase 4 review item 1: every
    token ties on cost, so only lexicographic order among surfaces can break
    the tie, and there can be many surface-distinct segmentations tied at the
    same minimum cost.
    """

    entries = []
    alphabet = "abcdefghij"
    for start in range(len(reading)):
        for end in range(start + 1, min(start + 3, len(reading)) + 1):
            span_reading = reading[start:end]
            for _ in range(rng.randint(1, 3)):
                surface = "".join(rng.choice(alphabet) for _ in range(end - start))
                entries.append(DictionaryEntry(span_reading, surface, "noun", 0.0))
    return Dictionary(entries)


@pytest.mark.parametrize("seed", range(12))
def test_top1_path_matches_brute_force_minimum_under_heavy_ties(seed: int) -> None:
    rng = random.Random(seed)
    reading = "".join(rng.choice("あいうえおか") for _ in range(6))
    dictionary = _random_tie_heavy_dictionary(rng, reading)
    model = scorer()
    graph = build_lattice(reading, dictionary)

    expected = _brute_force_minimum(graph, model)
    path = _top1_path(graph, model)

    assert expected is not None
    assert path is not None
    assert (path.cost, path.surface) == expected


def test_top1_path_matches_converters_own_reranked_head_under_ties() -> None:
    """The frontier search must agree with Converter's independent (cost,
    surface) sort even when many candidates tie, not just in the simple
    unweighted cases covered above."""

    rng = random.Random(1234)
    reading = "".join(rng.choice("あいうえお") for _ in range(5))
    dictionary = _random_tie_heavy_dictionary(rng, reading)
    model = scorer()
    converted = Converter(dictionary, beam_width=200, scorer=model).convert(
        reading, k=200, allow_literals=True
    )
    assert top1_surface(build_lattice(reading, dictionary), model) == converted.candidates[0].surface


def test_top1_path_is_fast_under_zero_cost_tie_explosion() -> None:
    """Regression for issue #9 Phase 4 review item 1 (performance).

    A reading whose every dictionary match ties at zero lexical cost used to
    make ``_top1_path`` re-run a bounded tie enumeration from scratch at
    probe sizes 8, 16, 32, ... up to 4096 -- 6.36s for a 12-character A2
    (``decomposed_v1-trans_only``) reading before this fix. The frontier
    search visits each lattice edge at most once, so it should stay well
    under a second even for a dictionary this ambiguous.
    """

    rng = random.Random(7)
    reading = "".join(rng.choice("あいうえお") for _ in range(12))
    entries = [
        DictionaryEntry(reading[start:end], "".join(rng.choice("abcdefgh") for _ in range(end - start)), "noun", 0.0)
        for start in range(len(reading))
        for end in range(start + 1, min(start + 4, len(reading)) + 1)
        for _ in range(4)
    ]
    dictionary = Dictionary(entries)
    model = scorer()
    graph = build_lattice(reading, dictionary)

    started = time.perf_counter()
    path = _top1_path(graph, model)
    elapsed = time.perf_counter() - started

    assert path is not None
    assert elapsed < 1.0, f"_top1_path took {elapsed:.3f}s under a zero-cost tie explosion"
