"""Exact-surface constrained best-path dynamic programming."""

from __future__ import annotations

from collections.abc import Hashable

from .lattice import Lattice, LatticePath, _path_key
from .scorer import Scorer


def constrained_best_path(
    lattice: Lattice,
    target_surface: str,
    scorer: Scorer,
) -> LatticePath | None:
    """Return the lowest-cost path whose surface is exactly ``target_surface``.

    The target surface constrains only edge selection; its boundaries need not
    match token boundaries.  Since the lattice is a DAG, one best path per
    ``(input position, target position, previous state)`` is sufficient.
    """

    paths: dict[tuple[int, int, Hashable | None], LatticePath] = {
        (0, 0, None): LatticePath(0.0),
    }

    for node in lattice.nodes:
        node_paths = tuple(
            (key, path)
            for key, path in paths.items()
            if key[0] == node.position
        )
        for (input_position, target_position, previous_state), path in node_paths:
            for edge in node.edges:
                surface = edge.token.surface
                if not target_surface.startswith(surface, target_position):
                    continue
                next_target_position = target_position + len(surface)
                edge_state = scorer.state_key(edge)
                next_key = (edge.end, next_target_position, edge_state)
                candidate = LatticePath(
                    path.cost
                    + scorer.edge_cost(edge)
                    + scorer.transition_cost(previous_state, edge),
                    edge.token,
                    path,
                    edge_state,
                )
                current = paths.get(next_key)
                if current is None or _path_key(candidate) < _path_key(current):
                    paths[next_key] = candidate

    terminal = len(lattice.nodes) - 1
    candidates = (
        path
        for (input_position, target_position, _state), path in paths.items()
        if input_position == terminal and target_position == len(target_surface)
    )
    return min(candidates, key=_path_key, default=None)


__all__ = ["constrained_best_path"]
