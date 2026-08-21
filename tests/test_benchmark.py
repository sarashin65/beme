"""Tests for AJIMEE sentence metrics and Mozc regression semantics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from beme.benchmark import (
    AjimeeCase,
    character_error_rate,
    convert_sentence,
    edit_distance,
    evaluate_ajimee,
    evaluate_mozc,
    katakana_to_hiragana,
    load_ajimee_cases,
    load_mozc_cases,
)
from beme.converter import Converter


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def write_dictionary(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "dictionary.tsv"
    path.write_text(
        "reading\tsurface\tpos\tcost\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_edit_distance_and_character_error_rate() -> None:
    assert edit_distance("家計を助ける", "家系を助ける") == 1
    assert edit_distance("abc", "ab") == 1
    assert edit_distance("ab", "acb") == 1
    assert character_error_rate("家計を助ける", "家系を助ける") == pytest.approx(1 / 6)


def test_sentence_adapter_normalizes_katakana_and_preserves_literals(tmp_path: Path) -> None:
    dictionary = write_dictionary(
        tmp_path,
        [
            "かけい\t家計\tnoun\t1.0",
            "たすける\t助ける\tverb\t1.0",
        ],
    )
    converter = Converter.from_tsv(dictionary)

    candidates = convert_sentence(converter, "カケイ、タスケル。", k=3)

    assert katakana_to_hiragana("カケイ、タスケル。") == "かけい、たすける。"
    assert candidates[0].surface == "家計、助ける。"


def test_ajimee_uses_best_permitted_reference(tmp_path: Path) -> None:
    dictionary = write_dictionary(tmp_path, ["かけい\t家計\tnoun\t1.0"])
    converter = Converter.from_tsv(dictionary)
    cases = [
        AjimeeCase(
            id="ajimee-test",
            context="",
            input="カケイ",
            references=("家系", "家計"),
        )
    ]

    report = evaluate_ajimee(converter, cases)

    assert report["case_count"] == 1
    assert report["mean_cer"] == 0.0
    assert report["mean_edit_distance"] == 0.0
    assert report["exact_match_accuracy"] == 1.0
    assert report["per_case"][0]["normalized_input"] == "かけい"


def test_mozc_conversion_commands_and_suggestion_skip(tmp_path: Path) -> None:
    dictionary = write_dictionary(
        tmp_path,
        [
            "これはうれしいごさんです\tこれは嬉しい誤算です\tnoun\t1.0",
            "きんじて\t禁じて\tverb\t1.0",
            "きんじて\t禁じる手\tverb\t2.0",
            "きんじて\t禁じ手\tnoun\t3.0",
        ],
    )
    cases_path = tmp_path / "mozc.tsv"
    cases_path.write_text(
        "# label\tkey\tvalue\tcommand\n"
        "issue-match\tこれはうれしいごさんです\t誤算\tConversion Match\n"
        "issue-negative\tあけましておめでとうございます\t開けまして\tConversion Not Match\n"
        "issue-rank\tきんじて\t禁じ手\tConversion Expected 2\n"
        "issue-suggestion\tぱわふる\tパワフル急速沸騰\tSuggestion Not Expected\n",
        encoding="utf-8",
    )

    report = evaluate_mozc(
        Converter.from_tsv(dictionary),
        load_mozc_cases(cases_path),
        k=3,
    )

    assert report["case_count"] == 4
    assert report["evaluated_count"] == 3
    assert report["passed_count"] == 3
    assert report["pass_rate"] == 1.0
    assert report["skipped_count"] == 1
    assert report["skipped_commands"] == {"Suggestion Not Expected": 1}
    assert report["per_case"][2]["expected_max_rank"] == 2


def test_vendored_evaluation_data_has_expected_step_one_size() -> None:
    ajimee = load_ajimee_cases(REPOSITORY_ROOT / "data" / "ajimee_bench_jwtd_v2_v1.json")
    mozc = load_mozc_cases(REPOSITORY_ROOT / "data" / "mozc_oss.tsv")

    assert len(ajimee) == 200
    assert len(mozc) == 500
    assert sum(case.command == "Suggestion Not Expected" for case in mozc) == 2
    assert sum(1 for _ in (REPOSITORY_ROOT / "data" / "evaluation_cases.jsonl").open(encoding="utf-8")) == 14


def test_evaluation_cli_runs_ajimee_suite(tmp_path: Path) -> None:
    dictionary = write_dictionary(tmp_path, ["かけい\t家計\tnoun\t1.0"])
    cases_path = tmp_path / "ajimee.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "index": "1",
                    "context_text": "",
                    "input": "カケイ",
                    "expected_output": ["家計"],
                    "original_text": "家計",
                    "splitted_input_for_limited_input_length": [],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPOSITORY_ROOT / "src"), env.get("PYTHONPATH", "")) if part
    )

    process = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "evaluate.py"),
            "--suite",
            "ajimee",
            "--dictionary",
            str(dictionary),
            "--ajimee-cases",
            str(cases_path),
            "--json",
        ],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["suite"] == "ajimee"
    assert report["mean_cer"] == 0.0
    assert report["exact_match_accuracy"] == 1.0



def test_evaluation_cli_reports_cross_encoder_fallback_without_model(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPOSITORY_ROOT / "src"), env.get("PYTHONPATH", "")) if part
    )
    process = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "evaluate.py"),
            "--suite",
            "handmade",
            "--reranker",
            "cross-encoder",
            "--cross-encoder-cache-dir",
            str(tmp_path / "empty-cache"),
            "--cross-encoder-local-files-only",
            "--json",
        ],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(process.stdout)

    assert process.returncode == 0, process.stderr
    assert report["reranker_requested"] == "cross-encoder"
    assert report["reranker_effective"] == "mock"
    assert report["cross_encoder_model"] is None
    assert "falling back to mock reranker" in process.stderr