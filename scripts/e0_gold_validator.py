"""Validation and deterministic repair of provisional E0 gold paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from beme.dictionary import Dictionary
from beme.lattice import LatticePath, build_lattice
from beme.oracle import constrained_best_path


class _CostScorer:
    """Small scorer used for repair; repair must not depend on a model score."""

    def state_key(self, edge: Any) -> str:
        return edge.token.pos

    def edge_cost(self, edge: Any) -> float:
        return edge.token.cost

    def transition_cost(self, previous: Any, edge: Any) -> float:
        return 0.0


def _token_dict(token: Any, reading_start: int, reading_end: int) -> dict[str, Any]:
    return {
        "surface": token.surface,
        "reading": token.reading,
        "surface_start": 0,
        "surface_end": len(token.surface),
        "reading_start": reading_start,
        "reading_end": reading_end,
        "is_literal": token.pos == "literal",
    }


def _path_dict(path: LatticePath) -> list[dict[str, Any]]:
    cursor = 0
    surface_cursor = 0
    result = []
    for token in path.tokens:
        end = cursor + len(token.reading)
        item = _token_dict(token, cursor, end)
        item["surface_start"] = surface_cursor
        item["surface_end"] = surface_cursor + len(token.surface)
        result.append(item)
        cursor = end
        surface_cursor += len(token.surface)
    return result


def _exact_dictionary_path(tokens: Sequence[Mapping[str, Any]], reading: str, dictionary: Dictionary) -> bool:
    """Check the supplied boundaries and lexical entries against the lattice."""
    cursor = 0
    for token in tokens:
        token_reading = str(token.get("reading", ""))
        surface = str(token.get("surface", ""))
        if not token_reading or not surface or reading[cursor:cursor + len(token_reading)] != token_reading:
            return False
        if bool(token.get("is_literal", False)):
            if surface != token_reading:
                return False
        elif not any(entry.surface == surface for entry in dictionary.entries_for_reading(token_reading)):
            return False
        cursor += len(token_reading)
    return cursor == len(reading)


def validate_gold_paths(candidates: Sequence[Mapping[str, Any]], dictionary: Dictionary) -> list[dict[str, Any]]:
    """Validate and, when possible, repair every candidate's provisional gold paths."""
    scorer = _CostScorer()
    output: list[dict[str, Any]] = []
    for original in candidates:
        candidate = dict(original)
        reading = str(candidate.get("input", ""))
        canonical = str(candidate.get("canonical_reference", ""))
        lattice = build_lattice(reading, dictionary, allow_literals=True)
        repaired_paths: list[dict[str, Any]] = []
        statuses: list[dict[str, Any]] = []
        raw_paths = list(candidate.get("gold_paths", []))
        if not raw_paths and candidate.get("gold_tokens"):
            raw_paths = [{"path_id": "canonical", "is_canonical": True,
                          "tokens": list(candidate["gold_tokens"])}]
        for raw_path in raw_paths:
            path = dict(raw_path)
            tokens = list(path.get("tokens", []))
            path_reading = "".join(str(token.get("reading", "")) for token in tokens)
            path_surface = "".join(str(token.get("surface", "")) for token in tokens)
            reading_valid = path_reading == reading
            surface_valid = path_surface == canonical
            reachable = reading_valid and surface_valid and _exact_dictionary_path(tokens, reading, dictionary)
            replacement = None
            if not reachable and surface_valid:
                replacement = constrained_best_path(lattice, canonical, scorer)
                if replacement is not None:
                    path["tokens"] = _path_dict(replacement)
                    path["repaired"] = True
            path["reading_valid"] = reading_valid
            path["surface_valid"] = surface_valid
            path["gold_reachable"] = reachable
            path["replacement_found"] = replacement is not None
            repaired_paths.append(path)
            statuses.append(path)
        candidate["gold_paths"] = repaired_paths
        candidate["gold_reading_valid"] = all(item["reading_valid"] for item in statuses) if statuses else False
        candidate["gold_surface_valid"] = all(item["surface_valid"] for item in statuses) if statuses else False
        candidate["gold_reachable"] = any(item["gold_reachable"] for item in statuses)
        candidate["gold_lattice_reachable"] = candidate["gold_reachable"]
        output.append(candidate)
    return output


def enrich_candidate(candidate: Mapping[str, Any], dictionary: Dictionary) -> dict[str, Any]:
    """Return one candidate with validated gold metadata."""
    return validate_gold_paths([candidate], dictionary)[0]


__all__ = ["enrich_candidate", "validate_gold_paths"]
