"""DAG lattice construction and deterministic exact K-best Viterbi."""

from __future__ import annotations

import functools
import math
import heapq
from itertools import count
from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable

from .dictionary import Dictionary, DictionaryEntry, is_hiragana

if TYPE_CHECKING:
    from collections.abc import Sequence
    from .scorer import Scorer


def _resolve_scorer(scorer: Scorer | None) -> Scorer:
    if scorer is not None:
        return scorer
    # Keep the import lazy: scorer implementations use lattice data classes.
    from .scorer import default_scorer

    return default_scorer()


UNKNOWN_COST = 8.0


@dataclass(frozen=True, slots=True)
class Token:
    """A lexical token carried by a lattice path."""

    reading: str
    surface: str
    pos: str
    cost: float
    pos_detail: str = ""
    conjugation: str = ""
    left_id: int = -1
    right_id: int = -1
    source: str = ""
    raw_cost: int | None = None

    @classmethod
    def from_entry(cls, entry: DictionaryEntry) -> "Token":
        return cls(
            entry.reading, entry.surface, entry.pos, float(entry.cost),
            entry.pos_detail, entry.conjugation, entry.left_id, entry.right_id,
            entry.source, entry.raw_cost,
        )


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
    state: Hashable | None = None

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

    def k_best(self, beam_width: int = 64, scorer: Scorer | None = None) -> tuple[LatticePath, ...]:
        return k_best_viterbi(self, beam_width=beam_width, scorer=scorer)

    def paths_at(
        self, position: int, beam_width: int = 64, scorer: Scorer | None = None
    ) -> tuple[LatticePath, ...]:
        """Return the retained paths ending at a node (useful for inspection)."""

        if not 0 <= position < len(self.nodes):
            raise IndexError(f"lattice position out of range: {position}")
        return _paths_by_node(self, beam_width, scorer)[position]

    def backward_costs(self, scorer: Scorer | None = None) -> tuple[float, ...]:
        """Return the minimum cost from each position to the terminal node."""

        return backward_viterbi(self, scorer=scorer)


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


# Compatibility view for callers of the old public helper.  The values live
# in the legacy scorer; this mapping deliberately contains no second table.
#
# Built lazily (see _transition_cost_table and module __getattr__ below)
# rather than at import time: legacy_v1_weights() opens and parses
# data/weights/legacy_v1.json, and a wheel install does not necessarily ship
# that repo-relative path (see weights.DEFAULT_WEIGHTS_PATH). Doing that I/O
# as a side effect of "import beme" made even a bare import fail once the
# package left the repo checkout -- a regression from before Phase 1
# (issue #9 Phase 1 review, M2).


@functools.lru_cache(maxsize=1)
def _transition_cost_table() -> tuple[dict[tuple[str, str], float], float, float, float]:
    """Return ``(pairs, trans.bos, trans.same_pos, trans.default)`` from the
    legacy weights file.  Cached: legacy_v1_weights() is only read once."""

    # Keep the import lazy, matching _resolve_scorer(): scorer implementations
    # use lattice data classes, and this keeps a bare "import beme" free of
    # file I/O regardless of import order.
    from .scorer import legacy_v1_weights

    weights = legacy_v1_weights()
    pairs: dict[tuple[str, str], float] = {}
    for name, value in weights.items():
        if name.startswith("trans.pair."):
            previous, current = name.removeprefix("trans.pair.").split(">", 1)
            pairs[(previous, current)] = value
    return pairs, weights["trans.bos"], weights["trans.same_pos"], weights["trans.default"]


def __getattr__(name: str) -> object:
    # PEP 562: keeps `from beme.lattice import _TRANSITION_COSTS` working
    # (tests/test_scorer.py) without paying for it at import time.
    if name == "_TRANSITION_COSTS":
        pairs, _bos, _same_pos, _default = _transition_cost_table()
        return pairs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def pos_transition_cost(previous_pos: str | None, current_pos: str) -> float:
    """Return the fixed Phase 1 POS-boundary cost, read from legacy weights.

    The first token has no BOS penalty (``trans.bos``).  Known natural
    boundaries use the pair table; same-POS and otherwise unseen transitions
    fall back to ``trans.same_pos`` and ``trans.default``.  All four values
    come from the same legacy_v1_weights() document that LinearScorer scores
    with -- there is no second, hardcoded source of truth for them.
    """

    pairs, bos, same_pos, default = _transition_cost_table()
    if previous_pos is None:
        return bos
    if (previous_pos, current_pos) in pairs:
        return pairs[(previous_pos, current_pos)]
    if previous_pos == current_pos:
        return same_pos
    return default


def backward_viterbi(lattice: Lattice, scorer: Scorer | None = None) -> tuple[float, ...]:
    """Compute minimum remaining path costs with a backward dynamic program.

    The returned value at a position treats the first edge leaving that
    position as a BOS transition.  POS transition costs between later edges
    are included.  Internally, the DP retains one continuation cost per POS,
    since a transition cost cannot be determined from the position alone.
    Unconnected positions have an infinite remaining cost.
    """

    scorer = _resolve_scorer(scorer)
    terminal = len(lattice.nodes) - 1
    if terminal < 0:
        return tuple()

    # continuation[position][previous_pos] is the cost after a token with
    # previous_pos has already been selected at that position.
    continuation: list[dict[Hashable, float]] = [{} for _ in lattice.nodes]
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
                        scorer.edge_cost(next_edge)
                        + scorer.transition_cost(scorer.state_key(edge), next_edge)
                        + next_costs[scorer.state_key(next_edge)]
                    )
                    for next_edge in lattice.nodes[edge.end].edges
                ) if next_costs else math.inf

            edge_cost = scorer.edge_cost(edge) + after_edge
            if edge_cost < best_from_node:
                best_from_node = edge_cost

            # Store only the cost after this edge.  A predecessor adds this
            # edge's token cost and transition cost itself.
            state = scorer.state_key(edge)
            continuation[node.position][state] = min(
                continuation[node.position].get(state, math.inf),
                after_edge,
            )
        node_costs[node.position] = best_from_node

    return tuple(node_costs)


def backward_costs_by_state(
    lattice: Lattice,
    scorer: Scorer,
    *,
    states: Sequence[Hashable] | None = None,
) -> tuple[dict[Hashable | None, float], ...]:
    """Compute exact remaining costs indexed by the scorer's continuation state.

    A transition scorer may depend on the previous edge only through
    ``state_key``.  Keeping the state domain to states entering each position
    keeps this DP bounded when a future scorer has a larger state vocabulary.
    """

    terminal = len(lattice.nodes) - 1
    if terminal < 0:
        return tuple()

    if states is None:
        domains: list[set[Hashable | None]] = [set() for _ in lattice.nodes]
        domains[0] = {None}
        for node in lattice.nodes:
            for edge in node.edges:
                domains[edge.end].add(scorer.state_key(edge))
    else:
        all_states = set(states)
        domains = [set(all_states) for _ in lattice.nodes]
        domains[0] = {None}

    remaining: list[dict[Hashable | None, float]] = [
        {state: math.inf for state in domain} for domain in domains
    ]
    for state in domains[terminal]:
        remaining[terminal][state] = 0.0

    for position in range(terminal - 1, -1, -1):
        for previous in domains[position]:
            best = math.inf
            for edge in lattice.nodes[position].edges:
                edge_state = scorer.state_key(edge)
                after_edge = remaining[edge.end].get(edge_state, math.inf)
                if math.isinf(after_edge):
                    continue
                best = min(
                    best,
                    scorer.transition_cost(previous, edge)
                    + scorer.edge_cost(edge)
                    + after_edge,
                )
            remaining[position][previous] = best

    return tuple(remaining)


def backward_costs_by_pos(lattice: Lattice) -> tuple[dict[str | None, float], ...]:
    """Compute exact remaining costs for every position and previous POS.

    ``R[position][previous_pos]`` is the minimum cost of completing the
    lattice from ``position`` when ``previous_pos`` is the POS of the token
    immediately before it.  Keeping the POS in the state is necessary because
    transition costs cannot be represented by a position-only heuristic.
    """

    scorer = _resolve_scorer(None)
    positions = {None}
    positions.update(scorer.state_key(edge) for edge in lattice.edges)
    return backward_costs_by_state(lattice, scorer, states=positions)


backward_costs = backward_viterbi
viterbi_backward = backward_viterbi


def _path_key(path: LatticePath) -> tuple[float, str, tuple[tuple[str, str, str, float], ...]]:
    return (
        path.cost,
        path.surface,
        tuple((token.reading, token.surface, token.pos, token.cost) for token in path.tokens),
    )


def _paths_by_node(
    lattice: Lattice, beam_width: int, scorer: Scorer | None = None
) -> tuple[tuple[LatticePath, ...], ...]:
    if not isinstance(beam_width, int) or isinstance(beam_width, bool) or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    scorer = _resolve_scorer(scorer)
    paths: list[list[LatticePath]] = [[] for _ in lattice.nodes]
    paths[0] = [LatticePath(0.0)]
    for node in lattice.nodes:
        if not paths[node.position]:
            continue
        for edge in node.edges:
            destination = paths[edge.end]
            for previous in paths[node.position]:
                previous_state = previous.state
                total = previous.cost + scorer.edge_cost(edge) + scorer.transition_cost(previous_state, edge)
                destination.append(LatticePath(total, edge.token, previous, scorer.state_key(edge)))
            destination.sort(key=_path_key)
            if len(destination) > beam_width:
                del destination[beam_width:]
    return tuple(tuple(sorted(node_paths, key=_path_key)) for node_paths in paths)


def k_best_viterbi(
    lattice: Lattice, beam_width: int = 64, scorer: Scorer | None = None
) -> tuple[LatticePath, ...]:
    """Return up to ``beam_width`` terminal paths in exact cost order.

    ``beam_width`` is retained as the public argument name for compatibility,
    but it is now only the requested number of results.  Each queued state is
    an entire persistent path.  The backward Viterbi cost is an admissible
    heuristic because all token and transition costs are non-negative, so a
    terminal state is safe to emit whenever it reaches the heap head.
    """

    if not isinstance(beam_width, int) or isinstance(beam_width, bool) or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")

    scorer = _resolve_scorer(scorer)
    remaining = backward_costs_by_state(lattice, scorer)
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
    max_pops = 200 * beam_width * len(lattice.nodes)
    pops = 0

    while queue and len(results) < beam_width:
        _, _, _, position, path = heapq.heappop(queue)
        pops += 1
        # This heuristic is strict, so the guard should not normally be reached;
        # it is a fail-safe for regressions that accidentally make search unbounded.
        if pops > max_pops:
            break
        if position == terminal:
            results.append(path)
            continue

        previous_state = path.state
        for edge in lattice.nodes[position].edges:
            edge_state = scorer.state_key(edge)
            after_edge = remaining[edge.end].get(edge_state, math.inf)
            if math.isinf(after_edge):
                continue
            cost = path.cost + scorer.edge_cost(edge) + scorer.transition_cost(previous_state, edge)
            child = LatticePath(cost, edge.token, path, edge_state)
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
    "backward_costs_by_state",
    "backward_viterbi",
    "k_best_viterbi",
    "pos_transition_cost",
    "viterbi_k_best",
    "viterbi_backward",
]
