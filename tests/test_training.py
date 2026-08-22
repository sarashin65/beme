"""Tests for the averaged structured perceptron core."""

from __future__ import annotations

from beme.features import FeatureExtractor, LEGACY_V1_SPEC
from beme.lattice import Lattice, LatticeEdge, LatticeNode, Token
from beme.scorer import LinearScorer
from beme.training import evaluate_split, evaluate_test, train_scorer
from beme.training_data import TrainingExample
from beme.weights import Weights


def graph() -> Lattice:
    return Lattice(
        "a",
        (
            LatticeNode(
                0,
                (
                    LatticeEdge(0, 1, Token("a", "A", "noun", 1.0)),
                    LatticeEdge(0, 1, Token("a", "B", "noun", 3.0)),
                ),
            ),
            LatticeNode(1, ()),
        ),
    )


def scorer() -> LinearScorer:
    return LinearScorer(
        FeatureExtractor(LEGACY_V1_SPEC),
        Weights({"lex.base_cost": 1.0, "trans.bos": 0.0}, feature_spec="legacy_v1"),
    )


def builder(_reading: str) -> Lattice:
    return graph()


def example(target: str) -> TrainingExample:
    return TrainingExample("a", target)


def test_one_cost_update_moves_gold_below_prediction() -> None:
    model = scorer()
    result = train_scorer([example("B")], [], builder, model, epochs=1, learning_rate=1.0)

    assert result.current_weights["lex.base_cost"] == -1.0
    assert result.averaged_weights["lex.base_cost"] == -1.0
    assert result.epochs[0].train.model_errors == 1
    assert evaluate_split([example("B")], builder, model).exact_matches == 1


def test_correct_examples_are_steps_but_coverage_errors_are_not() -> None:
    model = scorer()
    result = train_scorer(
        [example("A"), example("missing")], [], builder, model, epochs=1, learning_rate=1.0
    )

    train = result.epochs[0].train
    assert (train.examples, train.coverage_errors, train.covered_examples) == (2, 1, 1)
    assert train.model_errors == 0
    assert result.averaged_weights.metadata["averaging_steps"] == 1
    assert result.averaged_weights.metadata["updates"] == 0


def test_split_rates_and_zero_example_boundaries() -> None:
    metrics = evaluate_split([], builder, scorer())
    assert metrics.examples == metrics.coverage_errors == metrics.covered_examples == 0
    assert metrics.model_errors == metrics.end_to_end_errors == metrics.exact_matches == 0
    assert metrics.coverage_error_rate == metrics.model_error_rate == metrics.end_to_end_error_rate == 0.0

    metrics = evaluate_split([example("missing"), example("B")], builder, scorer())
    assert metrics.coverage_errors == 1
    assert metrics.covered_examples == 1
    assert metrics.model_errors == 1
    assert metrics.end_to_end_errors == 2
    assert metrics.coverage_error_rate == 0.5
    assert metrics.model_error_rate == 1.0
    assert metrics.end_to_end_error_rate == 1.0


def test_average_is_after_each_covered_example_and_gold_is_current_model() -> None:
    model = scorer()
    result = train_scorer(
        [example("B"), example("A")], [], builder, model, epochs=1, learning_rate=1.0, seed=4
    )

    # w=1 -> -1 after B update; B is then predicted, so A is a second update:
    # -1 + (3 - 1) = 1. The average of post-example weights is zero.
    assert result.current_weights["lex.base_cost"] == 1.0
    assert result.averaged_weights["lex.base_cost"] == 0.0
    assert result.epochs[0].train.model_errors == 2


def test_repeated_runs_and_dev_ties_are_deterministic() -> None:
    first = train_scorer([example("B")], [], builder, scorer(), epochs=2, learning_rate=1.0, seed=7)
    second = train_scorer([example("B")], [], builder, scorer(), epochs=2, learning_rate=1.0, seed=7)

    assert first.selected_epoch == second.selected_epoch == 1
    assert first.averaged_weights.to_dict() == second.averaged_weights.to_dict()


def test_test_evaluation_is_explicitly_after_selection() -> None:
    calls: list[str] = []

    def recording_builder(reading: str) -> Lattice:
        calls.append(reading)
        return graph()

    result = train_scorer([example("B")], [example("B")], recording_builder, scorer(), epochs=1, learning_rate=1.0)
    assert calls == ["a", "a"]  # train, then dev; no test was passed to training
    evaluation = evaluate_test(result, [example("B")], recording_builder, scorer())
    assert evaluation.examples == 1
    assert calls == ["a", "a", "a"]
