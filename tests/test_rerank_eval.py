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
from beme.rerank_stats import top1_from_scores
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
    combined = top1_from_scores(
        record["candidates"], viterbi_weight=1.0, reranker_weight=1.0, rerank_k=rerank_k,
    )
    assert combined == expected.candidates[0].surface


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
    assert result["pair"] == {
        "both_correct": 0, "one_correct": 0, "both_wrong": 0, "incomplete_pairs": 1,
    }


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


def _write_scores(path: Path, records: list[dict]) -> None:
    path.write_text(
        rerank_eval._jsonl_text([{"record_type": "provenance", "note": path.name}, *records]),
        encoding="utf-8",
    )


def test_combine_runs_each_scores_file_independently(tmp_path: Path) -> None:
    """Regression for issue #9 Phase 4 review item 3.

    `--scores a b` used to crash: after checking that `a` and `b` have
    identical id order, it appended `b`'s records onto `a`'s, guaranteeing
    duplicate ids that paired_table()/_top1_stats() then reject. A third
    file made it worse -- its id-order check compared against the
    already-extended list, so it could never match. `--scores` is documented
    (task2-mission.md) as "1 or more; multiple gives a labeled comparison",
    which independent per-file grids satisfy without ever needing a shared
    id set.
    """

    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    _write_scores(path_a, [scored_record()])
    _write_scores(path_b, [{**scored_record(), "id": "two"}])

    combined = tmp_path / "combined.json"
    assert rerank_eval.main([
        "combine", "--scores", str(path_a), str(path_b),
        "--viterbi-weight", "1", "--rerank-k", "2", "--output", str(combined),
    ]) == 0
    output = json.loads(combined.read_text(encoding="utf-8"))
    assert set(output) == {"runs"}
    assert len(output["runs"]) == 2
    assert {run["source"] for run in output["runs"]} == {str(path_a), str(path_b)}

    solo = tmp_path / "solo.json"
    assert rerank_eval.main([
        "combine", "--scores", str(path_a),
        "--viterbi-weight", "1", "--rerank-k", "2", "--output", str(solo),
    ]) == 0
    solo_output = json.loads(solo.read_text(encoding="utf-8"))
    run_a = next(run for run in output["runs"] if run["source"] == str(path_a))
    assert run_a["grid"] == solo_output["grid"]


def test_combine_accepts_three_or_more_scores_files(tmp_path: Path) -> None:
    paths = []
    for suffix in ("x", "y", "z"):
        path = tmp_path / f"{suffix}.jsonl"
        _write_scores(path, [{**scored_record(), "id": f"id-{suffix}"}])
        paths.append(path)
    output_path = tmp_path / "out.json"
    assert rerank_eval.main([
        "combine", "--scores", *[str(path) for path in paths],
        "--viterbi-weight", "1", "--rerank-k", "2", "--output", str(output_path),
    ]) == 0
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(output["runs"]) == 3


def test_latency_is_reported_once_per_run_not_once_per_grid_row(tmp_path: Path) -> None:
    """Regression for issue #9 Phase 4 review item 4.

    `latency` used to be recomputed identically inside every grid row even
    though it only reflects the cached `score` run, not that row's
    `viterbi_weight`/`rerank_k`/`gate` -- so it read as if latency varied per
    grid point when it never did.
    """

    source = tmp_path / "scores.jsonl"
    _write_scores(source, [scored_record()])
    output_path = tmp_path / "out.json"
    assert rerank_eval.main([
        "combine", "--scores", str(source),
        "--viterbi-weight", "1,2", "--rerank-k", "2,4", "--output", str(output_path),
    ]) == 0
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(output["grid"]) == 4
    assert "latency" in output
    assert all("latency" not in row for row in output["grid"])


def test_combine_gate_selects_by_each_records_own_confidence(tmp_path: Path) -> None:
    """Regression for issue #9 Phase 4 review item 5.

    The gate used to select records via `record in selected`, a deep dict
    membership check against a precomputed subset built with the same
    predicate -- O(n) per record and O(n**2) overall (1.30s vs .17s at
    2,000 x 64 in the review), for a result that only ever depends on the
    record's own `confidence`. This pins that per-record semantics survive
    the fix: a record below the gate gets reranked (and can flip to
    correct), a record above it keeps its original, cost-only order.
    """

    base = scored_record()
    ranked = {**base, "id": "ranked", "gold": "D", "confidence": 0.1}
    unranked = {**base, "id": "unranked", "gold": "D", "confidence": 0.9}
    result = rerank_eval.combine_records([ranked, unranked], 1.0, 4, gate=0.5)
    gained_ids = {item["id"] for item in result["all"]["gained"]}
    assert gained_ids == {"ranked"}
