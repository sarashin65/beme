"""Tests for exact-surface constrained oracle paths."""

from __future__ import annotations

from dataclasses import dataclass
from beme.lattice import Lattice, LatticeEdge, LatticeNode, LatticePath, Token, _path_key
from beme.oracle import constrained_best_path


@dataclass
class FakeScorer:
    edge_weights: dict[str, float]
    transition_weights: dict[tuple[object, str], float] | None = None

    def state_key(self, edge: LatticeEdge) -> tuple[str, str]:
        return ("custom", edge.token.pos)

    def edge_cost(self, edge: LatticeEdge) -> float:
        return self.edge_weights.get(edge.token.surface, edge.token.cost)

    def transition_cost(self, previous: object | None, edge: LatticeEdge) -> float:
        if self.transition_weights is None:
            return 0.0
        return self.transition_weights.get((previous, edge.token.surface), 0.0)


def edge(start: int, end: int, surface: str, *, reading: str | None = None, pos: str = "word") -> LatticeEdge:
    return LatticeEdge(start, end, Token(reading or surface, surface, pos, 0.0))


def lattice(*nodes: tuple[LatticeEdge, ...]) -> Lattice:
    return Lattice("ab", tuple(LatticeNode(position, edges) for position, edges in enumerate(nodes)))


def test_returns_unique_exact_surface_path() -> None:
    value = constrained_best_path(
        lattice((edge(0, 1, "X"),), (edge(1, 2, "Y"),), ()),
        "XY",
        FakeScorer({"X": 1.0, "Y": 2.0}),
    )

    assert value is not None
    assert (value.surface, value.reading, value.cost) == ("XY", "XY", 3.0)


def test_weight_change_selects_a_different_segmentation() -> None:
    graph = lattice(
        (edge(0, 1, "X"), edge(0, 2, "XY")),
        (edge(1, 2, "Y"),),
        (),
    )

    first = constrained_best_path(graph, "XY", FakeScorer({"X": 1, "Y": 1, "XY": 3}))
    second = constrained_best_path(graph, "XY", FakeScorer({"X": 2, "Y": 2, "XY": 1}))

    assert first is not None and second is not None
    assert first.tokens == (graph.nodes[0].edges[0].token, graph.nodes[1].edges[0].token)
    assert second.tokens == (graph.nodes[0].edges[1].token,)


def test_missing_target_returns_none() -> None:
    assert constrained_best_path(
        lattice((edge(0, 1, "X"),), (edge(1, 2, "Y"),), ()), "XZ", FakeScorer({})
    ) is None


def test_both_input_and_target_must_be_consumed() -> None:
    graph = lattice((edge(0, 1, "X"), edge(0, 2, "XY")), (edge(1, 2, "Y"),), ())

    assert constrained_best_path(graph, "X", FakeScorer({})) is None
    assert constrained_best_path(graph, "XYZ", FakeScorer({})) is None


def test_custom_hashable_state_is_used_for_transitions() -> None:
    graph = lattice((edge(0, 1, "X", pos="left"),), (edge(1, 2, "Y", pos="right"),), ())
    scorer = FakeScorer(
        {"X": 0.0, "Y": 0.0},
        {(None, "X"): 0.0, (("custom", "left"), "Y"): -4.0},
    )

    result = constrained_best_path(graph, "XY", scorer)

    assert result is not None
    assert result.cost == -4.0


def test_equal_cost_tie_does_not_depend_on_edge_insertion_order() -> None:
    first = edge(0, 1, "X", reading="b", pos="same")
    second = edge(0, 1, "X", reading="a", pos="same")
    graph_a = lattice((first, second), (edge(1, 2, "Y"),), ())
    graph_b = lattice((second, first), (edge(1, 2, "Y"),), ())

    result_a = constrained_best_path(graph_a, "XY", FakeScorer({}))
    result_b = constrained_best_path(graph_b, "XY", FakeScorer({}))

    assert result_a is not None and result_b is not None
    assert result_a.tokens == result_b.tokens
    assert result_a.tokens[0].reading == "a"


def test_negative_edge_and_transition_costs_are_supported() -> None:
    graph = lattice(
        (edge(0, 1, "X"), edge(0, 2, "XY")),
        (edge(1, 2, "Y"),),
        (),
    )
    scorer = FakeScorer(
        {"X": -1.0, "Y": -1.0, "XY": -1.5},
        {(None, "X"): -2.0, (None, "XY"): 0.0, (("custom", "word"), "Y"): -2.0},
    )

    result = constrained_best_path(graph, "XY", scorer)

    assert result is not None
    assert result.tokens == (graph.nodes[0].edges[0].token, graph.nodes[1].edges[0].token)
    assert result.cost == -6.0


def test_result_matches_exhaustive_exact_surface_enumeration() -> None:
    graph = lattice(
        (edge(0, 1, "A"), edge(0, 2, "AB")),
        (edge(1, 2, "B"),),
        (),
    )
    scorer = FakeScorer({"A": 3.0, "B": -1.0, "AB": 1.0})
    exhaustive: list[LatticePath] = []

    def visit(position: int, path: LatticePath) -> None:
        if position == len(graph.nodes) - 1:
            if path.surface == "AB":
                exhaustive.append(path)
            return
        for current in graph.nodes[position].edges:
            visit(
                current.end,
                LatticePath(
                    path.cost + scorer.edge_cost(current) + scorer.transition_cost(path.state, current),
                    current.token,
                    path,
                    scorer.state_key(current),
                ),
            )

    visit(0, LatticePath(0.0))
    result = constrained_best_path(graph, "AB", scorer)

    assert result is not None
    assert _path_key(result) == min(_path_key(path) for path in exhaustive)
