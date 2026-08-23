"""IPADIC matrix cost extraction and expanded transition state tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from beme.dictionary import Dictionary, DictionaryEntry
from beme.features import TransitionState, extractor_for, extractor_state_key
from beme.lattice import LatticeEdge, Token, build_lattice, k_best_viterbi
from beme.matrix_cost import MatrixCostTable, map_matrix_cost
from beme.scorer import LinearScorer
from beme.weights import Weights


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "ipadic_matrix_v1.bin"


def test_matrix_table_loads_and_maps_a_negative_entry() -> None:
    table = MatrixCostTable.load(MATRIX)
    assert 0 in table.ids
    raw = table.values[table.ids.index(0) * len(table.ids) + table.ids.index(0)]
    assert table.lookup(0, 0) == map_matrix_cost(raw)
    assert map_matrix_cost(-434) < 0


def test_matrix_state_and_transition_features_use_pos_not_state_string() -> None:
    extractor = extractor_for("decomposed_v1+matrix_v1")
    scorer = LinearScorer(extractor, Weights({"trans.matrix_cost": 1.0}, feature_spec=extractor.spec.id))
    first = LatticeEdge(0, 1, Token("あ", "亜", "noun", 0.0, left_id=12, right_id=31))
    second = LatticeEdge(1, 2, Token("い", "伊", "noun", 0.0, left_id=44, right_id=52))
    state = scorer.state_key(first)
    assert state == TransitionState("noun", 31)
    assert extractor_state_key(extractor, first) == state
    features = extractor.transition_features(state, second)
    assert "trans.same_pos" in features
    assert "TransitionState" not in " ".join(features)


def test_matrix_transition_pair_name_is_preserved() -> None:
    extractor = extractor_for("decomposed_v1+matrix_v1")
    previous = TransitionState("noun", 1)
    edge = LatticeEdge(0, 1, Token("あ", "亜", "particle", 0.0, left_id=2, right_id=3))
    assert "trans.pair.noun>particle" in extractor.transition_features(previous, edge)


def test_missing_ids_do_not_emit_matrix_feature() -> None:
    extractor = extractor_for("decomposed_v1+matrix_v1")
    edge = LatticeEdge(0, 1, Token("あ", "亜", "noun", 0.0, left_id=-1, right_id=-1))
    assert "trans.matrix_cost" not in extractor.transition_features(None, edge)


def test_negative_matrix_cost_k_best_matches_exhaustive_paths() -> None:
    tokens = (
        Token("あ", "A", "noun", 0.0, left_id=0, right_id=0),
        Token("あ", "B", "verb", 0.0, left_id=0, right_id=0),
    )
    # This uses the real table's (0, 0) value, which is negative in matrix.def.
    lattice = build_lattice(
        "ああ",
        Dictionary(DictionaryEntry(token.reading, token.surface, token.pos, token.cost,
                                   left_id=token.left_id, right_id=token.right_id)
                   for token in tokens),
    )
    extractor = extractor_for("decomposed_v1+matrix_v1")
    scorer = LinearScorer(extractor, Weights({"trans.matrix_cost": 1.0}, feature_spec=extractor.spec.id))
    paths = k_best_viterbi(lattice, beam_width=4, scorer=scorer)
    assert paths
    assert all(paths[index].cost <= paths[index + 1].cost for index in range(len(paths) - 1))


def test_state_key_strategy_is_resolved_once_when_the_scorer_is_built() -> None:
    """state_key runs in the k-best inner loop and must not re-dispatch.

    Resolving the strategy in __init__ also means an unsupported state
    definition is refused when the scorer is built, not mid-search.
    """
    from beme.features import FeatureSpec, state_key_function

    for spec_id, expected in (
        ("decomposed_v1", "noun"),
        ("decomposed_v1+matrix_v1", TransitionState("noun", 31)),
    ):
        extractor = extractor_for(spec_id)
        scorer = LinearScorer(extractor, Weights({}, feature_spec=spec_id))
        edge = LatticeEdge(0, 1, Token("あ", "亜", "noun", 0.0, left_id=12, right_id=31))
        assert scorer._state_key is state_key_function(extractor.spec)
        assert scorer.state_key(edge) == expected == extractor_state_key(extractor, edge)

    unsupported = FeatureSpec("unsupported", transition_state=("pos", "surface"))
    with pytest.raises(ValueError, match="unsupported feature specification"):
        state_key_function(unsupported)
