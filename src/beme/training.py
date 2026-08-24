"""Deterministic averaged structured perceptron training."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .features import FeatureExtractor, feature_delta, path_features
from .lattice import Lattice, LatticePath, _top1_path, top1_surface
from .oracle import constrained_best_path
from .scorer import LinearScorer
from .training_data import TrainingExample
from .weights import Weights


LatticeBuilder = Callable[[str], Lattice]


@dataclass(frozen=True, slots=True)
class SplitEvaluation:
    """Immutable error counts and rates for one data split."""

    examples: int
    coverage_errors: int
    covered_examples: int
    model_errors: int
    end_to_end_errors: int
    exact_matches: int
    coverage_error_rate: float
    model_error_rate: float
    end_to_end_error_rate: float

    @classmethod
    def from_counts(
        cls,
        *,
        examples: int,
        coverage_errors: int,
        covered_examples: int,
        model_errors: int,
        exact_matches: int,
    ) -> "SplitEvaluation":
        end_to_end_errors = coverage_errors + model_errors
        return cls(
            examples,
            coverage_errors,
            covered_examples,
            model_errors,
            end_to_end_errors,
            exact_matches,
            coverage_errors / examples if examples else 0.0,
            model_errors / covered_examples if covered_examples else 0.0,
            end_to_end_errors / examples if examples else 0.0,
        )


@dataclass(frozen=True, slots=True)
class EpochResult:
    """One training epoch and its dev evaluation under averaged weights."""

    epoch: int
    train: SplitEvaluation
    dev: SplitEvaluation
    averaged_weights: Weights


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Immutable training output; test evaluation is deliberately separate."""

    epochs: tuple[EpochResult, ...]
    selected_epoch: int
    current_weights: Weights
    averaged_weights: Weights


def _prediction(
    example: TrainingExample,
    lattice_builder: LatticeBuilder,
    scorer: LinearScorer,
) -> LatticePath | None:
    return _top1_path(lattice_builder(example.reading), scorer)


def evaluate_split(
    examples: Sequence[TrainingExample],
    lattice_builder: LatticeBuilder,
    scorer: LinearScorer,
) -> SplitEvaluation:
    """Evaluate coverage and model errors using the scorer's current weights."""

    coverage_errors = 0
    model_errors = 0
    exact_matches = 0
    for example in examples:
        lattice = lattice_builder(example.reading)
        prediction_surface = top1_surface(lattice, scorer)
        gold = constrained_best_path(lattice, example.target_surface, scorer)
        if gold is None:
            coverage_errors += 1
            continue
        if prediction_surface == example.target_surface:
            exact_matches += 1
        else:
            model_errors += 1
    return SplitEvaluation.from_counts(
        examples=len(examples),
        coverage_errors=coverage_errors,
        covered_examples=len(examples) - coverage_errors,
        model_errors=model_errors,
        exact_matches=exact_matches,
    )


def _updated_weights(scorer: LinearScorer, delta: Mapping[str, float], rate: float) -> Weights:
    # feature_delta is gold - pred; cost minimization therefore uses its
    # negation, w_cost += eta * (Phi_pred - Phi_gold).
    values = dict(scorer.weights.values)
    for name, value in delta.items():
        values[name] = values.get(name, scorer.weights.default_weight) - rate * value
    return Weights(values, scorer.weights.default_weight, scorer.weights.feature_spec)


def _averaged_weights(
    initial: Weights,
    totals: Mapping[str, float],
    steps: int,
    metadata: Mapping[str, Any],
) -> Weights:
    values = {
        name: value / steps for name, value in totals.items()
    } if steps else dict(initial.values)
    return Weights(values, initial.default_weight, initial.feature_spec, metadata)


def train_scorer(
    train_examples: Sequence[TrainingExample],
    dev_examples: Sequence[TrainingExample],
    lattice_builder: LatticeBuilder,
    scorer: LinearScorer,
    *,
    epochs: int,
    learning_rate: float,
    seed: int = 0,
    extractor: FeatureExtractor | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train with cost-minimizing averaged structured perceptron updates.

    Test examples are intentionally absent from this API.  Call
    :func:`evaluate_test` only after selecting an epoch from the returned
    dev snapshots.
    """

    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if learning_rate < 0:
        raise ValueError("learning_rate must be non-negative")
    if extractor is None:
        extractor = scorer.extractor
    initial = scorer.weights
    base_metadata = dict(metadata or {})
    totals: dict[str, float] = {}
    averaging_steps = 0
    updates = 0
    epoch_results: list[EpochResult] = []
    indices = list(range(len(train_examples)))

    for epoch in range(1, epochs + 1):
        random.Random(seed + epoch).shuffle(indices)
        train_coverage = 0
        train_model_errors = 0
        train_exact_matches = 0
        for index in indices:
            example = train_examples[index]
            lattice = lattice_builder(example.reading)
            prediction = _top1_path(lattice, scorer)
            gold = constrained_best_path(lattice, example.target_surface, scorer)
            if gold is None:
                train_coverage += 1
                continue

            averaging_steps += 1
            if prediction is not None and prediction.surface == example.target_surface:
                train_exact_matches += 1
            else:
                train_model_errors += 1
                if prediction is not None:
                    gold_features = path_features(gold, extractor, scorer.weights)
                    prediction_features = path_features(prediction, extractor, scorer.weights)
                    delta = feature_delta(gold_features, prediction_features)
                    new_weights = _updated_weights(scorer, delta, learning_rate)
                    scorer.invalidate(new_weights)
                    updates += 1
            for name in set(scorer.weights.values) | set(totals):
                if name not in totals:
                    # A feature can first appear mid-run.  Its effective value
                    # for the earlier steps was the default weight, so backfill
                    # them instead of averaging them in as zero.
                    totals[name] = (averaging_steps - 1) * initial.default_weight
                totals[name] += scorer.weights[name]

        train_evaluation = SplitEvaluation.from_counts(
            examples=len(train_examples),
            coverage_errors=train_coverage,
            covered_examples=len(train_examples) - train_coverage,
            model_errors=train_model_errors,
            exact_matches=train_exact_matches,
        )
        snapshot_metadata = {
            **base_metadata,
            "averaged": True,
            "updates": updates,
            "averaging_steps": averaging_steps,
            "epochs": epoch,
            "learning_rate": learning_rate,
            "seed": seed,
        }
        averaged = _averaged_weights(initial, totals, averaging_steps, snapshot_metadata)
        current = scorer.weights
        scorer.invalidate(averaged)
        try:
            dev_evaluation = evaluate_split(dev_examples, lattice_builder, scorer)
        finally:
            scorer.invalidate(current)
        epoch_results.append(EpochResult(epoch, train_evaluation, dev_evaluation, averaged))

    selected = min(
        epoch_results,
        key=lambda result: (
            result.dev.end_to_end_error_rate,
            result.dev.model_error_rate,
            result.epoch,
        ),
    )
    # The weights below come from the selected epoch, so their provenance
    # counters must come from that epoch too -- not from the end of the run.
    selected_metadata = dict(selected.averaged_weights.metadata or {})
    final_metadata = {
        **base_metadata,
        "averaged": True,
        "updates": selected_metadata["updates"],
        "averaging_steps": selected_metadata["averaging_steps"],
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "selected_epoch": selected.epoch,
    }
    final_averaged = Weights(
        dict(selected.averaged_weights.values),
        selected.averaged_weights.default_weight,
        selected.averaged_weights.feature_spec,
        final_metadata,
    )
    return TrainingResult(tuple(epoch_results), selected.epoch, scorer.weights, final_averaged)


def evaluate_test(
    result: TrainingResult,
    examples: Sequence[TrainingExample],
    lattice_builder: LatticeBuilder,
    scorer: LinearScorer,
) -> SplitEvaluation:
    """Evaluate held-out test data after dev model selection is complete."""

    current = scorer.weights
    scorer.invalidate(result.averaged_weights)
    try:
        return evaluate_split(examples, lattice_builder, scorer)
    finally:
        scorer.invalidate(current)


train = train_scorer
evaluate = evaluate_split


__all__ = [
    "EpochResult", "LatticeBuilder", "SplitEvaluation", "TrainingResult",
    "evaluate", "evaluate_split", "evaluate_test", "train", "train_scorer",
]
