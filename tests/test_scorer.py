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


def test_scorer_caches_reuse_entries_across_positions_instead_of_growing_unbounded() -> None:
    """Regression for issue #9 Phase 1 review M3.

    LinearScorer._edge_cache/_transition_cache used to key on the whole
    LatticeEdge, which carries the input position (start/end). Both feature
    extractors are documented as position independent (see
    features.path_features's docstring), so an edge that wraps an equal
    Token but sits at a different input position must hit the same cache
    entry. Before the fix these were two entries per distinct position
    instead of one per distinct token, which is why a converter that keeps
    reusing the same scorer across many conversions (scripts/evaluate.py; a
    long-lived IME process) grew both caches without bound -- 27,099 /
    169,847 entries for just 300 short mozc readings.
    """
    scorer = LinearScorer(FeatureExtractor(LEGACY_V1_SPEC), legacy_v1_weights())
    token = Token("か", "可", "noun", 1.25)
    near = LatticeEdge(0, 1, token)
    far = LatticeEdge(37, 38, token)  # same token, unrelated input position

    assert scorer.edge_cost(near) == scorer.edge_cost(far)
    assert len(scorer._edge_cache) == 1

    assert scorer.transition_cost("verb", near) == scorer.transition_cost("verb", far)
    assert len(scorer._transition_cache) == 1


def test_default_scorer_is_cached_process_wide() -> None:
    """Regression for issue #9 Phase 1 review L2.

    default_scorer() used to build a brand-new LinearScorer -- reparsing
    data/weights/legacy_v1.json through legacy_v1_weights() -- on every
    call. Every Lattice.k_best/backward_costs call and every Converter built
    without an explicit scorer goes through _resolve_scorer ->
    default_scorer(), so that reparsing sat on the conversion hot path.
    """
    import beme.scorer as scorer_module

    scorer_module._DEFAULT_SCORER = None  # isolate from whatever ran earlier
    first = default_scorer()
    second = default_scorer()
    assert first is second


def test_scorer_cache_memoization_does_not_change_computed_cost() -> None:
    """Memoization must be a pure optimization: a cache hit and a fresh
    computation for the same token must agree bit-for-bit."""
    extractor = FeatureExtractor(LEGACY_V1_SPEC)
    weights = legacy_v1_weights()
    memoized = LinearScorer(extractor, weights)
    fresh = LinearScorer(extractor, weights)
    token = Token("きしゃ", "汽車", "noun", 2.0)
    edge = LatticeEdge(0, 3, token)

    first = memoized.edge_cost(edge)
    second = memoized.edge_cost(edge)  # served from cache
    assert first == second == fresh.edge_cost(edge)
