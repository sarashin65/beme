"""Top-1 selection follows Converter's cost/surface tie rule."""

from __future__ import annotations

from beme.converter import Converter
from beme.dictionary import Dictionary, DictionaryEntry
from beme.lattice import Lattice, LatticeEdge, LatticeNode, Token, k_best_viterbi, top1_surface
from beme.scorer import LinearScorer
from beme.training import evaluate_split
from beme.features import FeatureExtractor, LEGACY_V1_SPEC
from beme.training_data import TrainingExample
from beme.weights import Weights


def scorer() -> LinearScorer:
    return LinearScorer(
        FeatureExtractor(LEGACY_V1_SPEC),
        Weights({"lex.base_cost": 1.0, "trans.bos": 0.0}, feature_spec="legacy_v1"),
    )


def lattice(surfaces: list[str]) -> Lattice:
    return Lattice(
        "あ",
        (
            LatticeNode(
                0,
                tuple(LatticeEdge(0, 1, Token("あ", surface, "noun", 1.0)) for surface in surfaces),
            ),
            LatticeNode(1, ()),
        ),
    )


def converter(surfaces: list[str]) -> Converter:
    return Converter(
        Dictionary(DictionaryEntry("あ", surface, "noun", 1.0) for surface in surfaces),
        beam_width=64,
        scorer=scorer(),
    )


def test_top1_matches_converter_on_equal_cost_paths() -> None:
    surfaces = ["ん", "あ", "う"]
    assert top1_surface(lattice(surfaces), scorer()) == "あ"
    assert converter(surfaces).convert("あ", k=1).candidates[0].surface == "あ"


def test_top1_matches_first_kbest_path_without_ties() -> None:
    graph = Lattice(
        "あ",
        (
            LatticeNode(0, (LatticeEdge(0, 1, Token("あ", "あ", "noun", 1.0)),)),
            LatticeNode(1, ()),
        ),
    )
    model = scorer()
    assert top1_surface(graph, model) == k_best_viterbi(graph, beam_width=1, scorer=model)[0].surface


def test_top1_returns_none_for_unconnected_lattice() -> None:
    graph = Lattice("あ", (LatticeNode(0, ()), LatticeNode(1, ())))
    assert top1_surface(graph, scorer()) is None


def test_top1_expands_probe_for_more_than_eight_ties() -> None:
    surfaces = [chr(ord("ぁ") + index) for index in range(9)]
    assert top1_surface(lattice(surfaces), scorer(), tie_probe=8) == min(surfaces)


def test_evaluate_split_uses_converter_top1_for_ties() -> None:
    surfaces = ["ん", "あ", "う"]
    model = scorer()
    builder = lambda _reading: lattice(surfaces)
    evaluation = evaluate_split([TrainingExample("あ", "あ")], builder, model)
    assert evaluation.exact_matches == 1
    assert evaluation.model_errors == 0
