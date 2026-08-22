"""Scorer interfaces and the fixed linear legacy scorer."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Protocol

from .weights import Weights

if TYPE_CHECKING:
    from .features import FeatureExtractor
    from .lattice import LatticeEdge


class Scorer(Protocol):
    spec_id: str

    def state_key(self, edge: LatticeEdge) -> Hashable: ...
    def edge_cost(self, edge: LatticeEdge) -> float: ...
    def transition_cost(self, previous: Hashable | None, edge: LatticeEdge) -> float: ...


def legacy_v1_weights() -> Weights:
    values = {
        "lex.base_cost": 1.0,
        "trans.bos": 0.0,
        "trans.pair.noun>particle": 0.05,
        "trans.pair.proper_noun>particle": 0.05,
        "trans.pair.pronoun>particle": 0.05,
        "trans.pair.particle>noun": 0.03,
        "trans.pair.particle>proper_noun": 0.03,
        "trans.pair.particle>verb": 0.04,
        "trans.pair.particle>adjective": 0.04,
        "trans.pair.verb>particle": 0.05,
        "trans.pair.adjective>auxiliary": 0.03,
        "trans.pair.verb>auxiliary": 0.03,
        "trans.pair.unknown>unknown": 0.20,
        "trans.pair.unknown>particle": 0.10,
        "trans.pair.particle>unknown": 0.10,
        "trans.same_pos": 0.08,
        "trans.default": 0.12,
    }
    return Weights(values)


class LinearScorer:
    """Score edges and boundaries as a weighted sum of extracted features."""

    def __init__(self, extractor: FeatureExtractor, weights: Weights) -> None:
        from .features import FeatureExtractor

        if not isinstance(extractor, FeatureExtractor):
            raise TypeError("extractor must be a FeatureExtractor")
        if not isinstance(weights, Weights):
            raise TypeError("weights must be Weights")
        self.extractor = extractor
        self.weights = weights
        self.spec_id = extractor.spec.id
        self._edge_cache: dict[LatticeEdge, float] = {}
        self._transition_cache: dict[tuple[Hashable | None, LatticeEdge], float] = {}

    def state_key(self, edge: LatticeEdge) -> Hashable:
        return edge.token.pos

    def _dot(self, features: dict[str, float]) -> float:
        total = 0.0
        for name, value in features.items():
            total += self.weights[name] * value
        return total

    def edge_cost(self, edge: LatticeEdge) -> float:
        if edge not in self._edge_cache:
            self._edge_cache[edge] = self._dot(self.extractor.lexical_features(edge))
        return self._edge_cache[edge]

    def transition_cost(self, previous: Hashable | None, edge: LatticeEdge) -> float:
        key = (previous, edge)
        if key not in self._transition_cache:
            self._transition_cache[key] = self._dot(self.extractor.transition_features(previous, edge))
        return self._transition_cache[key]

    def invalidate(self, weights: Weights | None = None) -> None:
        if weights is not None:
            if not isinstance(weights, Weights):
                raise TypeError("weights must be Weights")
            self.weights = weights
        self._edge_cache.clear()
        self._transition_cache.clear()


def default_scorer() -> LinearScorer:
    from .features import legacy_v1_extractor

    return LinearScorer(legacy_v1_extractor(), legacy_v1_weights())


__all__ = ["LinearScorer", "Scorer", "Weights", "default_scorer", "legacy_v1_weights"]
