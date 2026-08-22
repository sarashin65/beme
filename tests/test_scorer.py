"""L1 exact-equivalence tests for the legacy scorer."""

from __future__ import annotations

import pytest

from beme.features import FeatureExtractor, LEGACY_V1_SPEC
from beme.lattice import LatticeEdge, Token, _TRANSITION_COSTS, pos_transition_cost
from beme.scorer import LinearScorer, default_scorer, legacy_v1_weights


POS = [
    "adjective", "adnominal", "adverb", "auxiliary", "conjunction", "counter",
    "interjection", "noun", "number", "other", "particle", "prefix", "pronoun",
    "proper_noun", "suffix", "verb", "unknown", "literal",
]


@pytest.mark.parametrize("previous", [None, *POS])
@pytest.mark.parametrize("current", POS)
def test_legacy_transition_is_bit_exact(previous: str | None, current: str) -> None:
    edge = LatticeEdge(0, 1, Token("か", "可", current, 1.25))
    assert default_scorer().transition_cost(previous, edge) == pos_transition_cost(previous, current)


def test_legacy_edge_cost_is_bit_exact() -> None:
    scorer = default_scorer()
    edge = LatticeEdge(0, 1, Token("か", "可", "noun", 1.25))
    assert scorer.edge_cost(edge) == edge.token.cost


def test_legacy_features_have_one_active_feature_each() -> None:
    extractor = FeatureExtractor(LEGACY_V1_SPEC)
    edge = LatticeEdge(0, 1, Token("か", "可", "noun", 1.25))
    assert len(extractor.lexical_features(edge)) == 1
    assert len(extractor.transition_features(None, edge)) == 1
    assert len(extractor.transition_features("noun", edge)) == 1


def test_legacy_transition_weights_match_table() -> None:
    weights = legacy_v1_weights()
    assert {pair: weights[f"trans.pair.{pair[0]}>{pair[1]}"] for pair in _TRANSITION_COSTS} == _TRANSITION_COSTS
