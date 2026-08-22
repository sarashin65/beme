"""Tests for hand-computable IR metrics and aggregate evaluation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from beme.converter import Converter
from beme.evaluation import evaluate, load_cases, mrr_at_k, ndcg_at_k
from beme.reranker import MockContextReranker


def write_dictionary(tmp_path: Path) -> Path:
    path = tmp_path / "dictionary.tsv"
    path.write_text(
        "reading\tsurface\tpos\tcost\n"
        "かみ\t神\tnoun\t0\n"
        "かみ\t紙\tnoun\t1\n"
        "ねこ\t猫\tnoun\t0\n",
        encoding="utf-8",
    )
    return path


def test_ndcg_and_mrr_match_hand_calculation() -> None:
    judgments = {"A": 3, "B": 2, "C": 1}
    retrieved = ["B", "unjudged", "A"]
    ideal_dcg = 7 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    expected_ndcg = (3 / math.log2(2) + 7 / math.log2(4)) / ideal_dcg

    assert ndcg_at_k(retrieved, judgments, 3) == pytest.approx(expected_ndcg)
    assert mrr_at_k(retrieved, judgments, 3) == pytest.approx(1.0)
    assert mrr_at_k(["unjudged", "also-unjudged"], judgments, 2) == 0.0


def test_ndcg_is_zero_when_no_relevant_result_is_retrieved() -> None:
    assert ndcg_at_k(["x", "y"], {"A": 2}, 2) == 0.0
    assert ndcg_at_k(["x"], {}, 1) == 0.0


def test_evaluate_reports_exact_aggregate_metrics_and_rerank_rate(tmp_path: Path) -> None:
    dictionary = write_dictionary(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps({"id": "kami", "reading": "かみ", "judgments": {"神": 2}}, ensure_ascii=False)
        + "\n"
        + json.dumps({"id": "neko", "reading": "ねこ", "judgments": {"犬": 1}}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    report = evaluate(
        Converter.from_tsv(dictionary),
        load_cases(cases_path),
        k=2,
        cascade_threshold=2.0,
    )

    assert report["query_count"] == 2
    assert report["ndcg_at_k"] == pytest.approx(0.5)
    assert report["mrr_at_k"] == pytest.approx(0.5)
    assert report["top1_accuracy"] == pytest.approx(0.5)
    assert report["rerank_rate"] == pytest.approx(0.5)
    assert len(report["per_query"]) == 2


def test_contextual_cases_are_included_in_ndcg_and_mrr(tmp_path: Path) -> None:
    dictionary = tmp_path / "context-dictionary.tsv"
    dictionary.write_text(
        "reading\tsurface\tpos\tcost\n"
        "こうしょう\t交渉\tnoun\t0.0\n"
        "こうしょう\t考証\tnoun\t0.4\n",
        encoding="utf-8",
    )
    cases_path = tmp_path / "context-cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "history",
                "reading": "こうしょう",
                "preceding_context": "歴史を研究する",
                "judgments": {"考証": 3},
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps(
            {
                "id": "business",
                "reading": "こうしょう",
                "following_context": "について商談する",
                "judgments": {"交渉": 3},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate(
        Converter.from_tsv(dictionary, reranker=MockContextReranker()),
        load_cases(cases_path),
        k=2,
    )

    assert report["ndcg_at_k"] == pytest.approx(1.0)
    assert report["mrr_at_k"] == pytest.approx(1.0)
    assert report["top1_accuracy"] == pytest.approx(1.0)
    assert report["context_rerank_rate"] == pytest.approx(1.0)
    assert [query["retrieved"][0] for query in report["per_query"]] == ["考証", "交渉"]


def test_evaluation_script_emits_json_report(tmp_path: Path) -> None:
    dictionary = write_dictionary(tmp_path)
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps({"id": "kami", "reading": "かみ", "judgments": {"神": 1}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root / "src"), env.get("PYTHONPATH", "")) if part
    )

    process = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "evaluate.py"),
            "--dictionary",
            str(dictionary),
            "--cases",
            str(cases_path),
            "--top-k",
            "2",
            "--json",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["query_count"] == 1
    assert report["ndcg_at_k"] == pytest.approx(1.0)
    assert report["mrr_at_k"] == pytest.approx(1.0)


def test_evaluation_script_rejects_non_object_weights_json_without_traceback(tmp_path: Path) -> None:
    """Regression for issue #9 Phase 1 review L1.

    scripts/evaluate.py mirrored the CLI's self-referential --weights
    pre-read (json.load(stream).get("feature_spec", ...)), which raised an
    uncaught AttributeError for a top-level JSON array instead of the clean
    "error: ..." message the surrounding except (OSError, ValueError,
    TypeError) is meant to produce.
    """
    weights_path = tmp_path / "weights.json"
    weights_path.write_text("[]", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root / "src"), env.get("PYTHONPATH", "")) if part
    )

    process = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "evaluate.py"),
            "--weights",
            str(weights_path),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode != 0
    assert "traceback" not in process.stderr.lower()
    assert "error" in (process.stderr + process.stdout).lower()


def test_load_cases_rejects_malformed_or_nonpositive_judgments(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(malformed)

    invalid_judgment = tmp_path / "invalid-judgment.jsonl"
    invalid_judgment.write_text(
        json.dumps({"id": "bad", "reading": "かみ", "judgments": {"神": 0}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_cases(invalid_judgment)

    invalid_context = tmp_path / "invalid-context.jsonl"
    invalid_context.write_text(
        json.dumps(
            {
                "id": "bad-context",
                "reading": "かみ",
                "preceding_context": 1,
                "judgments": {"神": 1},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="preceding_context"):
        load_cases(invalid_context)


def test_cli_top_k_and_rerank_k_are_independent_widths(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.tsv"
    dictionary.write_text(
        "reading\tsurface\tpos\tcost\n"
        "かみ\t神\tnoun\t0\n"
        "かみ\t紙\tnoun\t1\n"
        "かみ\t髪\tnoun\t2\n"
        "かみ\t上\tnoun\t3\n",
        encoding="utf-8",
    )
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "kami",
                "reading": "かみ",
                "preceding_context": "神社で",
                "judgments": {"神": 1},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root / "src"), env.get("PYTHONPATH", "")) if part
    )

    def run(*extra: str) -> dict:
        process = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "evaluate.py"),
                "--dictionary",
                str(dictionary),
                "--cases",
                str(cases_path),
                *extra,
                "--json",
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode == 0, process.stderr
        return json.loads(process.stdout)

    # Reranking a wider set than the output width: only expressible once the
    # two widths stopped sharing a flag.
    wide = run("--top-k", "2", "--rerank-k", "3")
    assert wide["top_k"] == 2
    assert wide["rerank_k"] == 3
    assert wide["k"] == 2, "--top-k alone governs the metric cutoff"
    assert wide["scored_candidate_count"] == 3, "--rerank-k alone governs reranker input"
    assert len(wide["per_query"][0]["retrieved"]) == 2

    # Omitting --rerank-k keeps the historical behaviour: every candidate is
    # scored, and --top-k only trims the output.
    default = run("--top-k", "2")
    assert default["rerank_k"] is None
    assert default["k"] == 2
    assert default["scored_candidate_count"] == 4
    assert len(default["per_query"][0]["retrieved"]) == 2
