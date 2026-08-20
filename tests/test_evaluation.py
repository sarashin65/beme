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
