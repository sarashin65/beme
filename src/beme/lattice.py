"""DAG lattice construction and deterministic beam K-best Viterbi."""

from __future__ import annotations

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
    """A partial or complete path retained by beam Viterbi."""

    cost: float
    tokens: tuple[Token, ...]

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
    ) -> "Lattice":
        return build_lattice(reading, dictionary, unknown_cost=unknown_cost)

    def k_best(self, beam_width: int = 64) -> tuple[LatticePath, ...]:
        return k_best_viterbi(self, beam_width=beam_width)

    def paths_at(self, position: int, beam_width: int = 64) -> tuple[LatticePath, ...]:
        """Return the retained paths ending at a node (useful for inspection)."""

        if not 0 <= position < len(self.nodes):
            raise IndexError(f"lattice position out of range: {position}")
        return _paths_by_node(self, beam_width)[position]


def _edge_key(edge: LatticeEdge) -> tuple[int, float, str, str, str]:
    return (edge.end, edge.token.cost, edge.token.surface, edge.token.pos, edge.token.reading)


def build_lattice(
    reading: str,
    dictionary: Dictionary,
    *,
    unknown_cost: float = UNKNOWN_COST,
) -> Lattice:
    """Build a connected lattice, adding a one-character fallback if needed."""

    if not reading or not is_hiragana(reading):
        raise ValueError("reading must be non-empty hiragana (ー allowed)")
    if unknown_cost < 0:
        raise ValueError("unknown_cost must be non-negative")

    nodes: list[LatticeNode] = []
    for position in range(len(reading) + 1):
        if position == len(reading):
            nodes.append(LatticeNode(position, tuple()))
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
    paths[0] = [LatticePath(0.0, tuple())]
    for node in lattice.nodes:
        if not paths[node.position]:
            continue
        for edge in node.edges:
            destination = paths[edge.end]
            for previous in paths[node.position]:
                previous_pos = previous.tokens[-1].pos if previous.tokens else None
                total = previous.cost + edge.token.cost + pos_transition_cost(previous_pos, edge.token.pos)
                destination.append(LatticePath(total, previous.tokens + (edge.token,)))
            destination.sort(key=_path_key)
            if len(destination) > beam_width:
                del destination[beam_width:]
    return tuple(tuple(sorted(node_paths, key=_path_key)) for node_paths in paths)


def k_best_viterbi(lattice: Lattice, beam_width: int = 64) -> tuple[LatticePath, ...]:
    """Return the terminal node's deterministic beam-retained paths."""

    return _paths_by_node(lattice, beam_width)[-1]


viterbi_k_best = k_best_viterbi

__all__ = [
    "Lattice",
    "LatticeEdge",
    "LatticeNode",
    "LatticePath",
    "Token",
    "UNKNOWN_COST",
    "build_lattice",
    "k_best_viterbi",
    "pos_transition_cost",
    "viterbi_k_best",
]
