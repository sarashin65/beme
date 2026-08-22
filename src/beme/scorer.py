"""Scorer interfaces and the fixed linear legacy scorer."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Protocol

from .weights import DEFAULT_WEIGHTS_PATH, Weights

if TYPE_CHECKING:
    from .features import FeatureExtractor
    from .lattice import LatticeEdge, Token


class Scorer(Protocol):
    spec_id: str

    def state_key(self, edge: LatticeEdge) -> Hashable: ...
    def edge_cost(self, edge: LatticeEdge) -> float: ...
    def transition_cost(self, previous: Hashable | None, edge: LatticeEdge) -> float: ...


def legacy_v1_weights() -> Weights:
    return Weights.load(DEFAULT_WEIGHTS_PATH)


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
        # Keyed by edge.token, not by the LatticeEdge itself: lexical_features
        # and transition_features are documented as position-independent (see
        # features.path_features), so edge.start/edge.end carry no scoring
        # information -- they only fragmented the cache into one entry per
        # (position, token) pair instead of one per distinct token. A
        # long-lived Converter that keeps reusing the same scorer across many
        # conversions (scripts/evaluate.py; an IME process) never evicts
        # anything, so that fragmentation grew the cache without bound
        # (issue #9 Phase 1 review, M3). Keying by the token bounds it by the
        # number of distinct tokens ever scored -- for a fixed dictionary,
        # that is the dictionary plus a handful of literal/unknown variants.
        self._edge_cache: dict[Token, float] = {}
        self._transition_cache: dict[tuple[Hashable | None, Token], float] = {}

    def state_key(self, edge: LatticeEdge) -> Hashable:
        return edge.token.pos

    def _dot(self, features: dict[str, float]) -> float:
        total = 0.0
        for name, value in features.items():
            total += self.weights[name] * value
        return total

    def edge_cost(self, edge: LatticeEdge) -> float:
        token = edge.token
        if token not in self._edge_cache:
            self._edge_cache[token] = self._dot(self.extractor.lexical_features(edge))
        return self._edge_cache[token]

    def transition_cost(self, previous: Hashable | None, edge: LatticeEdge) -> float:
        key = (previous, edge.token)
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
