"""Tests for the Phase 4 score/combine harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import replace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import rerank_eval  # noqa: E402
from beme.converter import Converter
from beme.dictionary import Dictionary, DictionaryEntry
from beme.features import FeatureExtractor, LEGACY_V1_SPEC
from beme.reranker import ConversionContext, MockContextReranker
from beme.scorer import LinearScorer
from beme.training_data import TrainingExample
from beme.weights import Weights


def model() -> LinearScorer:
    return LinearScorer(
        FeatureExtractor(LEGACY_V1_SPEC),
        Weights({"lex.base_cost": 1.0, "trans.bos": 0.0}, feature_spec="legacy_v1"),
    )


def converter() -> Converter:
    dictionary = Dictionary(
        DictionaryEntry("あ", surface, "noun", cost)
        for surface, cost in (("A", 1.0), ("B", 1.2), ("C", 1.4), ("D", 1.6))
    )
    return Converter(dictionary, beam_width=4, scorer=model())


class FixedReranker:
    def __init__(self, relevance: dict[str, float]) -> None:
        self.relevance = relevance

    def rerank(self, candidates, context):
        return tuple(
            sorted(
                (replace(candidate, score=candidate.cost - self.relevance[candidate.surface]) for candidate in candidates),
                key=lambda candidate: (candidate.score, candidate.cost, candidate.surface),
            )
        )


def scored_record() -> dict:
    base = converter().convert("あ", k=4, allow_literals=True).candidates
    relevance = {"A": 0.0, "B": 0.9, "C": 0.1, "D": 2.0}
    ranked = FixedReranker(relevance).rerank(base, ConversionContext("文脈"))
    return {
        "id": "one", "reading": "あ", "gold": "B", "contrast_pair_id": "p",
        "context_empty": False, "candidates": [
            {"surface": candidate.surface, "cost": candidate.cost,
             "relevance": relevance[candidate.surface]}
            for candidate in base
        ], "elapsed_seconds": 1.0,
    }


@pytest.mark.parametrize("rerank_k", [None, 2, 10])
def test_combine_matches_converter_for_all_head_sizes(rerank_k: int | None) -> None:
    record = scored_record()
    relevance = {item["surface"]: item["relevance"] for item in record["candidates"]}
    actual = converter()
    actual.reranker = FixedReranker(relevance)
    actual.rerank_k = rerank_k
    expected = actual.convert("あ", k=4, context=ConversionContext("文脈"), allow_literals=True)
    combined = rerank_eval.top1_from_scores(record["candidates"], 1.0, 1.0, rerank_k)
    assert [item["surface"] for item in combined] == [candidate.surface for candidate in expected.candidates]


def test_empty_context_is_not_scored_and_keeps_order() -> None:
    examples = [TrainingExample("あ", "A", id="empty")]
    records = rerank_eval.score_examples(examples, converter(), MockContextReranker(), "mock")
    assert records[0]["context_empty"] is True
    assert all(candidate["relevance"] is None for candidate in records[0]["candidates"])
    result = rerank_eval.combine_records(records, 0.0, 1)
    assert result["context_empty"] == 1
    assert result["all"]["after_accuracy"] == result["all"]["accuracy"]


def test_mock_relevance_is_keyword_bonus() -> None:
    example = TrainingExample("あ", "考証", id="mock", left_context="歴史")
    dictionary = Dictionary([DictionaryEntry("あ", "考証", "noun", 1.0)])
    records = rerank_eval.score_examples([example], Converter(dictionary, beam_width=1, scorer=model()), MockContextReranker(), "mock")
    assert records[0]["candidates"][0]["relevance"] == 1.0


def test_absent_gold_is_counted_in_rank_distribution() -> None:
    record = scored_record()
    record["gold"] = "missing"
    result = rerank_eval.combine_records([record], 1.0, 2)
    assert result["all"]["gold_rank"]["absent"] == 1
    assert result["all"]["examples"] == 1


def test_incomplete_pair_is_explicitly_counted() -> None:
    record = scored_record()
    result = rerank_eval.combine_records([record], 1.0, 2)
    assert result["pair"] == {"both_correct": 0, "one_correct": 0, "both_wrong": 0, "incomplete": 1}


def test_combine_output_is_byte_stable(tmp_path: Path) -> None:
    source = tmp_path / "scores.jsonl"
    source.write_text(rerank_eval._jsonl_text([{"record_type": "provenance", "x": 1}, scored_record()]), encoding="utf-8")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert rerank_eval.main(["combine", "--scores", str(source), "--viterbi-weight", "1", "--rerank-k", "2", "--output", str(first)]) == 0
    assert rerank_eval.main(["combine", "--scores", str(source), "--viterbi-weight", "1", "--rerank-k", "2", "--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()


def test_score_schema_is_shared_by_all_ranker_kinds() -> None:
    example = TrainingExample("あ", "A", id="schema", left_context="文脈")
    for kind in ("mock", "cross-encoder", "llm"):
        records = rerank_eval.score_examples([example], converter(), MockContextReranker(), kind)
        assert set(records[0]) >= {"id", "reading", "gold", "candidates", "elapsed_seconds"}
        assert set(records[0]["candidates"][0]) == {"surface", "cost", "relevance"}
