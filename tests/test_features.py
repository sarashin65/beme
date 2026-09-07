"""Path feature extraction and serialization primitives."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from beme.dictionary import Dictionary
from beme.features import (
    FeatureExtractor,
    FeatureSpec,
    decomposed_v1_extractor,
    feature_delta,
    legacy_v1_extractor,
    path_features,
)
from beme.sparse_weights import SparseLexicalWeights
from beme.lattice import LatticeEdge, Token, build_lattice, k_best_viterbi
from beme.scorer import LinearScorer
from beme.weights import DEFAULT_WEIGHTS_PATH, Weights


def test_feature_delta_is_sparse_symmetric_and_sorted() -> None:
    gold = {"z": 3.0, "a": 2.0, "zero": 1.0}
    pred = {"z": 1.0, "a": 2.0, "zero": 1.0, "b": 4.0}
    assert list(feature_delta(gold, pred)) == ["b", "z"]
    assert feature_delta(gold, pred) == {"b": -4.0, "z": 2.0}
    assert feature_delta(gold, pred) == {
        key: -value for key, value in feature_delta(pred, gold).items()
    }


def test_path_features_reconstruct_legacy_cost() -> None:
    dictionary = Dictionary.from_tsv(Path("data/dictionary.tsv"))
    extractor = legacy_v1_extractor()
    scorer = LinearScorer(extractor, Weights.load(DEFAULT_WEIGHTS_PATH))
    for reading in ("きしゃ", "きしゃのきしゃ", "きしゃ123かな!"):
        lattice = build_lattice(reading, dictionary, allow_literals=True)
        for path in k_best_viterbi(lattice, beam_width=5, scorer=scorer):
            features = path_features(path, extractor)
            assert sum(scorer.weights[name] * value for name, value in features.items()) == path.cost


def test_decomposed_features_include_declared_lexical_signals() -> None:
    extractor = decomposed_v1_extractor()
    overlay = LatticeEdge(
        0, 2, Token("にほん", "日本", "proper_noun", 0.65, source="overlay")
    )
    ipadic = LatticeEdge(
        0, 2, Token("きしゃ", "汽車", "noun", 2.0, raw_cost=3500, source="ipadic")
    )

    overlay_features = extractor.lexical_features(overlay)
    ipadic_features = extractor.lexical_features(ipadic)
    assert overlay_features["lex.source.overlay"] == 1.0
    assert overlay_features["lex.overlay_cost"] == 0.65
    assert overlay_features["lex.reading_len"] == 3.0
    assert overlay_features["lex.surface_len"] == 2.0
    assert overlay_features["lex.script.kanji"] == 1.0
    assert overlay_features["lex.proper_noun"] == 1.0
    assert ipadic_features["lex.source.overlay"] == 0.0
    assert ipadic_features["lex.reading_len"] == 3.0
    assert ipadic_features["lex.surface_len"] == 2.0
    assert ipadic_features["lex.script.kanji"] == 1.0


def test_sparse_lexical_feature_uses_reading_surface_backoff() -> None:
    spec = FeatureSpec("test_sparse", ("lex.sparse_lexical",), family="legacy_v1")
    sparse = SparseLexicalWeights.empty()
    sparse.set_surface("汽車", 2.5)
    extractor = FeatureExtractor(spec, sparse_weights=sparse)
    edge = LatticeEdge(0, 3, Token("きしゃ", "汽車", "noun", 1.0))
    assert extractor.lexical_features(edge) == {"lex.sparse_lexical": 2.5}


def test_decomposed_ipadic_cost_mapped_honors_non_default_build_params() -> None:
    """Regression for issue #9 Phase 1 review M4.

    The decomposed extractor used to recompute
    ``map_ipadic_cost(token.raw_cost)`` with the module's default
    floor/ceiling constants, ignoring whatever
    --raw-cost-floor/--raw-cost-ceiling/--beme-cost-floor/--beme-cost-ceiling
    scripts/build_ipadic_dictionary.py actually built the dictionary with (or
    any later scripts/apply_cost_learning.py adjustment). Craft a token whose
    stored cost is what a non-default --beme-cost-floor would have produced,
    and check the decomposed features reconstruct that stored cost exactly
    rather than the default-parameter cost.
    """
    from beme.ipadic import map_ipadic_cost

    raw_cost = 5000
    custom_cost = map_ipadic_cost(raw_cost, beme_floor=1.0, beme_ceiling=9.0)
    default_cost = map_ipadic_cost(raw_cost)
    assert custom_cost != default_cost  # sanity: the two parameterizations disagree

    token = Token("きしゃ", "汽車", "noun", custom_cost, raw_cost=raw_cost, source="ipadic")
    edge = LatticeEdge(0, 3, token)
    extractor = decomposed_v1_extractor()

    features = extractor.lexical_features(edge)
    assert "lex.katakana_surface" not in features
    assert features["lex.ipadic_cost_mapped"] == pytest.approx(custom_cost)
    assert features["lex.ipadic_cost_mapped"] != pytest.approx(default_cost)


def test_decomposed_katakana_surface_matches_ipadic_build_predicate() -> None:
    """Regression for issue #9 Phase 1 review L3.

    The old inline check (``"ァ" <= char <= "ヺ"``, U+30A1-U+30FA only)
    missed the ranges ``ipadic.contains_katakana_surface`` also treats as
    katakana at dictionary build time (U+30FD-U+30FF, U+31F0-U+31FF). A
    surface in one of those ranges carries KATAKANA_SURFACE_COST_PENALTY in
    its token.cost, so failing to fire lex.katakana_surface for it would
    break the decomposed reconstruction.
    """
    from beme.ipadic import KATAKANA_SURFACE_COST_PENALTY, map_ipadic_cost

    raw_cost = 5000
    mapped = map_ipadic_cost(raw_cost)
    surface = "ヽ"  # katakana iteration mark: outside U+30A1-U+30FA
    token = Token(
        "あ", surface, "noun", mapped + KATAKANA_SURFACE_COST_PENALTY, raw_cost=raw_cost, source="ipadic",
    )
    edge = LatticeEdge(0, 1, token)
    extractor = decomposed_v1_extractor()

    features = extractor.lexical_features(edge)
    assert features["lex.katakana_surface"] == 1.0
    assert features["lex.ipadic_cost_mapped"] == pytest.approx(mapped)


def test_feature_json_round_trip(tmp_path: Path) -> None:
    document = {"features": {"lex.base_cost": 0.8, "trans.bos": 1.0}, "rank": 0}
    path = tmp_path / "features.jsonl"
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0]) == document


def test_path_features_reconstruct_cost_under_non_default_weights() -> None:
    """Regression: the residual correction must use the scoring weights.

    ``path.cost`` comes from whatever ``LinearScorer`` produced the path.
    Verifying it against the default weights file turns the whole weight
    difference into a bogus ``lex.base_cost`` term, which would corrupt every
    phase-2 perceptron update once training moves the weights.
    """

    dictionary = Dictionary.from_tsv(Path("data/dictionary.tsv"))
    extractor = legacy_v1_extractor()
    default = Weights.load(DEFAULT_WEIGHTS_PATH)
    trained = Weights(
        {**default.values, "lex.base_cost": 1.5, "trans.default": 0.30},
        default.default_weight,
        default.feature_spec,
    )
    scorer = LinearScorer(extractor, trained)

    lattice = build_lattice("きしゃのきしゃ", dictionary, allow_literals=True)
    paths = k_best_viterbi(lattice, beam_width=5, scorer=scorer)
    assert paths

    for path in paths:
        features = path_features(path, extractor, trained)
        # Exact reconstruction is a unit-weight property; with a trained
        # lex.base_cost the residual carry only restores it to rounding.
        assert sum(trained[name] * value for name, value in features.items()) == pytest.approx(path.cost)
        # The true unweighted token-cost sum survives the residual correction.
        assert features["lex.base_cost"] == pytest.approx(
            sum(token.cost for token in path.tokens)
        )

    # Verifying against the wrong weights is a programming error, not a
    # silently absorbed residual.
    with pytest.raises(ValueError, match="does not match the supplied weights"):
        path_features(paths[0], extractor, default)
