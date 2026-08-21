"""DAG lattice construction and deterministic exact K-best Viterbi."""

from __future__ import annotations

import math
import heapq
from itertools import count
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .dictionary import Dictionary, DictionaryEntry, is_hiragana

if TYPE_CHECKING:
    from collections.abc import Sequence


UNKNOWN_COST = 8.0


@dataclass(frozen=True, slots=True)
class Token:
    """A lexical token carried by a lattice path."""

    reading: str
    surface: str
    pos: str
    cost: float

    @classmethod
    def from_entry(cls, entry: DictionaryEntry) -> "Token":
        return cls(entry.reading, entry.surface, entry.pos, float(entry.cost))


@dataclass(frozen=True, slots=True)
class LatticeEdge:
    """A directed token edge from one input position to another."""

    start: int
    end: int
    token: Token

    @property
    def entry(self) -> Token:
        """Compatibility alias for callers that call edge payloads entries."""

        return self.token


@dataclass(frozen=True, slots=True)
class LatticeNode:
    """One position in the input-position DAG."""

    position: int
    edges: tuple[LatticeEdge, ...]

    @property
    def outgoing(self) -> tuple[LatticeEdge, ...]:
        return self.edges


@dataclass(frozen=True, slots=True)
class LatticePath:
    """A persistent linked-list path retained by K-best Viterbi."""

    cost: float
    token: Token | None = None
    parent: "LatticePath | None" = None

    @property
    def tokens(self) -> tuple[Token, ...]:
        """Materialize the path tokens only when the path is inspected."""

        reversed_tokens: list[Token] = []
        path: LatticePath | None = self
        while path is not None and path.token is not None:
            reversed_tokens.append(path.token)
            path = path.parent
        return tuple(reversed(reversed_tokens))

    @property
    def surface(self) -> str:
        return "".join(token.surface for token in self.tokens)

    @property
    def reading(self) -> str:
        return "".join(token.reading for token in self.tokens)


@dataclass(frozen=True, slots=True)
class Lattice:
    """A connected DAG whose nodes correspond to reading positions ``0..n``."""

    reading: str
    nodes: tuple[LatticeNode, ...]

    @property
    def edges(self) -> tuple[LatticeEdge, ...]:
        return tuple(edge for node in self.nodes for edge in node.edges)

    @classmethod
    def from_dictionary(
        cls,
        reading: str,
        dictionary: Dictionary,
        *,
        unknown_cost: float = UNKNOWN_COST,
        allow_literals: bool = True,
    ) -> "Lattice":
        return build_lattice(
            reading,
            dictionary,
            unknown_cost=unknown_cost,
            allow_literals=allow_literals,
        )

    def k_best(self, beam_width: int = 64) -> tuple[LatticePath, ...]:
        return k_best_viterbi(self, beam_width=beam_width)

    def paths_at(self, position: int, beam_width: int = 64) -> tuple[LatticePath, ...]:
        """Return the retained paths ending at a node (useful for inspection)."""

        if not 0 <= position < len(self.nodes):
            raise IndexError(f"lattice position out of range: {position}")
        return _paths_by_node(self, beam_width)[position]

    def backward_costs(self) -> tuple[float, ...]:
        """Return the minimum cost from each position to the terminal node."""

        return backward_viterbi(self)


def _edge_key(edge: LatticeEdge) -> tuple[int, float, str, str, str]:
    return (edge.end, edge.token.cost, edge.token.surface, edge.token.pos, edge.token.reading)


def build_lattice(
    reading: str,
    dictionary: Dictionary,
    *,
    unknown_cost: float = UNKNOWN_COST,
    allow_literals: bool = True,
) -> Lattice:
    """Build a connected lattice, including fixed edges for literals.

    Hiragana positions use dictionary edges and an unknown fallback.  Every
    other character is carried through as a zero-cost one-character edge so a
    sentence can be searched without introducing span boundaries.
    """

    if not isinstance(reading, str) or not reading:
        raise ValueError("reading must be a non-empty string")
    if (
        not allow_literals
        or any(0x30A1 <= ord(char) <= 0x30FA for char in reading)
    ) and not is_hiragana(reading):
        raise ValueError("reading must be non-empty hiragana (ー allowed)")
    if unknown_cost < 0:
        raise ValueError("unknown_cost must be non-negative")

    nodes: list[LatticeNode] = []
    for position in range(len(reading) + 1):
        if position == len(reading):
            nodes.append(LatticeNode(position, tuple()))
            continue
        if not is_hiragana(reading[position]):
            char = reading[position]
            edges = (
                LatticeEdge(
                    position,
                    position + 1,
                    Token(char, char, "literal", 0.0),
                ),
            )
            nodes.append(LatticeNode(position, edges))
            continue
        matches = dictionary.matches_at(reading, position)
        if matches:
            edges = tuple(
                sorted(
                    (
                        LatticeEdge(
                            position,
                            position + len(entry.reading),
                            Token.from_entry(entry),
                        )
                        for entry in matches
                    ),
                    key=_edge_key,
                )
            )
        else:
            char = reading[position]
            edges = (
                LatticeEdge(
                    position,
                    position + 1,
                    Token(char, char, "unknown", float(unknown_cost)),
                ),
            )
        nodes.append(LatticeNode(position, edges))
    return Lattice(reading, tuple(nodes))


# The transition table is deliberately tiny: Phase 1 has no learned model.
# Values are additive costs and stay small relative to lexical costs.  The
# default discourages implausible boundaries while leaving unknown fallback
# paths viable.
_TRANSITION_COSTS: dict[tuple[str, str], float] = {
    ("noun", "particle"): 0.05,
    ("proper_noun", "particle"): 0.05,
    ("pronoun", "particle"): 0.05,
    ("particle", "noun"): 0.03,
    ("particle", "proper_noun"): 0.03,
    ("particle", "verb"): 0.04,
    ("particle", "adjective"): 0.04,
    ("verb", "particle"): 0.05,
    ("adjective", "auxiliary"): 0.03,
    ("verb", "auxiliary"): 0.03,
    ("unknown", "unknown"): 0.20,
    ("unknown", "particle"): 0.10,
    ("particle", "unknown"): 0.10,
}


def pos_transition_cost(previous_pos: str | None, current_pos: str) -> float:
    """Return the fixed Phase 1 POS-boundary cost.

    The first token has no BOS penalty.  Known natural boundaries use the
    small table above; same-POS and otherwise unseen transitions receive 0.08
    and 0.12 respectively.
    """

    if previous_pos is None:
        return 0.0
    if (previous_pos, current_pos) in _TRANSITION_COSTS:
        return _TRANSITION_COSTS[(previous_pos, current_pos)]
    if previous_pos == current_pos:
        return 0.08
    return 0.12


def backward_viterbi(lattice: Lattice) -> tuple[float, ...]:
    """Compute minimum remaining path costs with a backward dynamic program.

    The returned value at a position treats the first edge leaving that
    position as a BOS transition.  POS transition costs between later edges
    are included.  Internally, the DP retains one continuation cost per POS,
    since a transition cost cannot be determined from the position alone.
    Unconnected positions have an infinite remaining cost.
    """

    terminal = len(lattice.nodes) - 1
    if terminal < 0:
        return tuple()

    # continuation[position][previous_pos] is the cost after a token with
    # previous_pos has already been selected at that position.
    continuation: list[dict[str, float]] = [{} for _ in lattice.nodes]
    node_costs = [math.inf] * len(lattice.nodes)
    node_costs[terminal] = 0.0

    for node in reversed(lattice.nodes[:-1]):
        best_from_node = math.inf
        for edge in node.edges:
            if edge.end == terminal:
                after_edge = 0.0
            else:
                next_costs = continuation[edge.end]
                after_edge = min(
                    (
                        next_edge.token.cost
                        + pos_transition_cost(edge.token.pos, next_edge.token.pos)
                        + next_costs[next_edge.token.pos]
                    )
                    for next_edge in lattice.nodes[edge.end].edges
                ) if next_costs else math.inf

            edge_cost = edge.token.cost + after_edge
            if edge_cost < best_from_node:
                best_from_node = edge_cost

            # Store only the cost after this edge.  A predecessor adds this
            # edge's token cost and transition cost itself.
            continuation[node.position][edge.token.pos] = min(
                continuation[node.position].get(edge.token.pos, math.inf),
                after_edge,
            )
        node_costs[node.position] = best_from_node

    return tuple(node_costs)


def backward_costs_by_pos(lattice: Lattice) -> tuple[dict[str | None, float], ...]:
    """Compute exact remaining costs for every position and previous POS.

    ``R[position][previous_pos]`` is the minimum cost of completing the
    lattice from ``position`` when ``previous_pos`` is the POS of the token
    immediately before it.  Keeping the POS in the state is necessary because
    transition costs cannot be represented by a position-only heuristic.
    """

    terminal = len(lattice.nodes) - 1
    if terminal < 0:
        return tuple()

    positions: set[str | None] = {None}
    for node in lattice.nodes:
        positions.update(edge.token.pos for edge in node.edges)

    remaining: list[dict[str | None, float]] = [{} for _ in lattice.nodes]
    for previous_pos in positions:
        remaining[terminal][previous_pos] = 0.0

    for position in range(terminal - 1, -1, -1):
        current = remaining[position]
        for previous_pos in positions:
            best = math.inf
            for edge in lattice.nodes[position].edges:
                after_edge = remaining[edge.end].get(edge.token.pos, math.inf)
                if math.isinf(after_edge):
                    continue
                best = min(
                    best,
                    pos_transition_cost(previous_pos, edge.token.pos)
                    + edge.token.cost
                    + after_edge,
                )
            current[previous_pos] = best

    return tuple(remaining)


backward_costs = backward_viterbi
viterbi_backward = backward_viterbi


def _path_key(path: LatticePath) -> tuple[float, str, tuple[tuple[str, str, str, float], ...]]:
    return (
        path.cost,
        path.surface,
        tuple((token.reading, token.surface, token.pos, token.cost) for token in path.tokens),
    )


def _paths_by_node(lattice: Lattice, beam_width: int) -> tuple[tuple[LatticePath, ...], ...]:
    if not isinstance(beam_width, int) or isinstance(beam_width, bool) or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    paths: list[list[LatticePath]] = [[] for _ in lattice.nodes]
    paths[0] = [LatticePath(0.0)]
    for node in lattice.nodes:
        if not paths[node.position]:
            continue
        for edge in node.edges:
            destination = paths[edge.end]
            for previous in paths[node.position]:
                previous_pos = previous.token.pos if previous.token is not None else None
                total = previous.cost + edge.token.cost + pos_transition_cost(previous_pos, edge.token.pos)
                destination.append(LatticePath(total, edge.token, previous))
            destination.sort(key=_path_key)
            if len(destination) > beam_width:
                del destination[beam_width:]
    return tuple(tuple(sorted(node_paths, key=_path_key)) for node_paths in paths)


def k_best_viterbi(lattice: Lattice, beam_width: int = 64) -> tuple[LatticePath, ...]:
    """Return up to ``beam_width`` terminal paths in exact cost order.

    ``beam_width`` is retained as the public argument name for compatibility,
    but it is now only the requested number of results.  Each queued state is
    an entire persistent path.  The backward Viterbi cost is an admissible
    heuristic because all token and transition costs are non-negative, so a
    terminal state is safe to emit whenever it reaches the heap head.
    """

    if not isinstance(beam_width, int) or isinstance(beam_width, bool) or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")

    remaining = backward_costs_by_pos(lattice)
    if not remaining or math.isinf(remaining[0].get(None, math.inf)):
        return tuple()

    # The counter is only a final tie breaker; the preceding fields make the
    # result independent of dictionary insertion order.
    sequence = count()
    queue: list[tuple[float, float, int, int, LatticePath]] = []
    root = LatticePath(0.0)
    heapq.heappush(queue, (remaining[0][None], 0.0, next(sequence), 0, root))
    results: list[LatticePath] = []
    terminal = len(lattice.nodes) - 1

    while queue and len(results) < beam_width:
        _, _, _, position, path = heapq.heappop(queue)
        if position == terminal:
            results.append(path)
            continue

        previous_pos = path.token.pos if path.token is not None else None
        for edge in lattice.nodes[position].edges:
            after_edge = remaining[edge.end].get(edge.token.pos, math.inf)
            if math.isinf(after_edge):
                continue
            cost = path.cost + edge.token.cost + pos_transition_cost(previous_pos, edge.token.pos)
            child = LatticePath(cost, edge.token, path)
            heapq.heappush(
                queue,
                (
                    cost + after_edge,
                    cost,
                    next(sequence),
                    edge.end,
                    child,
                ),
            )

    return tuple(results)


viterbi_k_best = k_best_viterbi

__all__ = [
    "Lattice",
    "LatticeEdge",
    "LatticeNode",
    "LatticePath",
    "Token",
    "UNKNOWN_COST",
    "build_lattice",
    "backward_costs",
    "backward_costs_by_pos",
    "backward_viterbi",
    "k_best_viterbi",
    "pos_transition_cost",
    "viterbi_k_best",
    "viterbi_backward",
]
