"""Contract tests for Phase 4's shared reranking statistics."""

from __future__ import annotations

from dataclasses import replace

import pytest

from beme.rerank_stats import (
    gold_rank_histogram,
    label,
    mcnemar,
    paired_table,
    pair_level,
    ranked_candidates_from_scores,
    stratify,
    top1_from_scores,
)
from beme.converter import Candidate
from beme.reranker import ConversionContext


def correctness(*values: tuple[str, bool]) -> list[dict[str, object]]:
    return [{"id": identifier, "correct": correct} for identifier, correct in values]


def test_paired_table_is_order_independent_and_rejects_id_mismatch_and_duplicates() -> None:
    before = correctness(("a", True), ("b", True), ("c", False), ("d", False))
    after = correctness(("d", False), ("c", True), ("b", False), ("a", True))
    assert paired_table(before, after) == {"a": 1, "b": 1, "c": 1, "d": 1}
    with pytest.raises(ValueError, match="id sets differ"):
        paired_table(before, after[:-1])
    with pytest.raises(ValueError, match="duplicate id"):
        paired_table(before + [before[0]], before)


def test_paired_table_self_comparison_has_no_discordant_cases() -> None:
    records = correctness(("right", True), ("wrong", False))
    assert paired_table(records, records) == {"a": 1, "b": 0, "c": 0, "d": 1}


@pytest.mark.parametrize(
    ("b", "c", "method"),
    [(1, 23, "exact_binomial_two_sided"), (1, 24, "continuity_corrected_chi_squared")],
)
def test_mcnemar_switches_method_at_25_discordant(b: int, c: int, method: str) -> None:
    assert mcnemar(b, c)["method"] == method


def test_mcnemar_phase3_golden_values() -> None:
    assert mcnemar(0, 0)["p_value"] == 1.0
    assert mcnemar(0, 30)["p_value"] == pytest.approx(1.1924366854416816e-07)


def test_label_requires_multiple_significant_seeds_with_one_direction() -> None:
    better = [{"a": 0, "b": 0, "c": 30, "d": 0} for _ in range(3)]
    worse = [{"a": 0, "b": 30, "c": 0, "d": 0} for _ in range(3)]
    assert label(better) == "significant_better"
    assert label(worse) == "significant_worse"
    assert label(better[:1]) == "indeterminate"
    assert label([better[0], {"a": 0, "b": 6, "c": 14, "d": 0}]) == "indeterminate"
    assert label([better[0], worse[0]]) == "indeterminate"


def test_label_has_no_api_for_a_pooled_table() -> None:
    with pytest.raises(TypeError, match="do not pool seeds"):
        label({"a": 0, "b": 0, "c": 90, "d": 0})  # type: ignore[arg-type]


class FixedReranker:
    def __init__(self, relevance: dict[str, float], viterbi_weight: float, reranker_weight: float) -> None:
        self.relevance = relevance
        self.viterbi_weight = viterbi_weight
        self.reranker_weight = reranker_weight

    def rerank(self, candidates, context):
        return tuple(sorted(
            (
                replace(
                    candidate,
                    score=self.viterbi_weight * candidate.cost
                    - self.reranker_weight * self.relevance[candidate.surface],
                )
                for candidate in candidates
            ),
            key=lambda candidate: (candidate.score, candidate.cost, candidate.surface),
        ))


@pytest.mark.parametrize("rerank_k", [None, 2, 10])
def test_score_order_and_top1_match_converter_reranker_contract(rerank_k: int | None) -> None:
    candidates = [Candidate("C", 1.0, ()), Candidate("B", 2.0, ()), Candidate("A", 3.0, ())]
    relevance = {"C": 0.0, "B": 2.0, "A": 4.0}
    head = candidates if rerank_k is None else candidates[:rerank_k]
    expected = list(FixedReranker(relevance, 1.0, 1.0).rerank(head, ConversionContext("文脈")))
    if rerank_k is not None:
        expected += candidates[rerank_k:]
    records = [
        {"surface": candidate.surface, "cost": candidate.cost, "relevance": relevance[candidate.surface]}
        for candidate in candidates
    ]
    actual = ranked_candidates_from_scores(
        records, viterbi_weight=1.0, reranker_weight=1.0, rerank_k=rerank_k,
    )
    assert [item["surface"] for item in actual] == [candidate.surface for candidate in expected]
    assert top1_from_scores(
        records, viterbi_weight=1.0, reranker_weight=1.0, rerank_k=rerank_k,
    ) == expected[0].surface


def test_score_ties_use_cost_then_surface_and_null_relevance_keeps_order() -> None:
    cost_tie = [
        {"surface": "高", "cost": 2.0, "relevance": 2.0},
        {"surface": "安", "cost": 1.0, "relevance": 1.0},
    ]
    assert top1_from_scores(
        cost_tie, viterbi_weight=1.0, reranker_weight=1.0, rerank_k=None,
    ) == "安"
    surface_tie = [
        {"surface": "ん", "cost": 1.0, "relevance": 1.0},
        {"surface": "あ", "cost": 1.0, "relevance": 1.0},
    ]
    assert top1_from_scores(
        surface_tie, viterbi_weight=1.0, reranker_weight=1.0, rerank_k=None,
    ) == "あ"
    mixed = [surface_tie[0], {**surface_tie[1], "relevance": None}]
    assert top1_from_scores(
        mixed, viterbi_weight=0.0, reranker_weight=1.0, rerank_k=None,
    ) == "ん"


def test_stratify_reading_length_has_an_overflow_bucket_past_sixteen() -> None:
    """Regression for issue #9 Phase 4 review item 7.

    `stratify(records, "reading_length")` used to raise for any reading
    longer than 16 characters, aborting the entire stratification instead of
    bucketing the outlier -- real readings are not capped at 16.
    """

    records = [
        {"reading_length": 17, "before_correct": True, "after_correct": True},
        {"reading_length": 40, "before_correct": False, "after_correct": True},
    ]
    layers = stratify(records, "reading_length")
    assert set(layers) == {"1-3", "4-6", "7-10", "11-16", "17+"}
    assert layers["17+"] == {
        "n": 2, "before_error_rate": 0.5, "after_error_rate": 0.0, "low_support": True,
    }
    for empty_bucket in ("1-3", "4-6", "7-10", "11-16"):
        assert layers[empty_bucket]["n"] == 0


def test_stratify_still_rejects_a_non_positive_reading_length() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        stratify(
            [{"reading_length": 0, "before_correct": True, "after_correct": True}],
            "reading_length",
        )


def test_stratify_emits_empty_fixed_layers_and_low_support() -> None:
    records = [{
        "reading_length": 2, "contrast_pair_id": "pair", "span_tokens": 1,
        "proper_noun": True, "before_correct": False, "after_correct": True,
    }]
    layers = stratify(records, "reading_length")
    assert layers["1-3"] == {
        "n": 1, "before_error_rate": 1.0, "after_error_rate": 0.0, "low_support": True,
    }
    assert layers["4-6"] == {
        "n": 0, "before_error_rate": None, "after_error_rate": None, "low_support": True,
    }
    assert stratify(records * 30, "proper_noun")["true"]["low_support"] is False
    assert set(stratify(records, "contrast_pair")) == {"false", "true"}
    assert set(stratify(records, "span_tokens")) == {"1"}


def test_pair_level_excludes_incomplete_pairs_from_outcomes() -> None:
    records = [
        {"contrast_pair_id": "both", "correct": True},
        {"contrast_pair_id": "both", "correct": True},
        {"contrast_pair_id": "one", "correct": True},
        {"contrast_pair_id": "one", "correct": False},
        {"contrast_pair_id": "wrong", "correct": False},
        {"contrast_pair_id": "wrong", "correct": False},
        {"contrast_pair_id": "incomplete", "correct": True},
    ]
    assert pair_level(records) == {
        "both_correct": 1, "one_correct": 1, "both_wrong": 1, "incomplete_pairs": 1,
    }


def test_gold_rank_histogram_counts_absent_in_total() -> None:
    records = [
        {"gold": "A", "candidates": [{"surface": "A"}]},
        {"gold": "missing", "candidates": [{"surface": "A"}]},
        {"gold": "last", "candidates": [str(index) for index in range(21)] + ["last"]},
    ]
    histogram = gold_rank_histogram(records)
    assert histogram["1"] == 1
    assert histogram[">20"] == 1
    assert histogram["absent"] == 1
    assert sum(histogram.values()) == len(records)
