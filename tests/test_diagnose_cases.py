"""Tests for deterministic case diagnosis output."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _run(output: Path) -> dict[str, object]:
    subprocess.run(
        [sys.executable, "scripts/diagnose_cases.py", "--suite", "handmade",
         "--dictionary", "data/dictionary_v2.tsv",
         "--baseline-weights", "data/weights/decomposed_v1.json",
         "--learned-weights", "data/weights/decomposed_v1.json",
         "--output", str(output)], cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_handmade_contains_all_cases_and_zero_diff(tmp_path: Path) -> None:
    document = _run(tmp_path / "diagnosis.json")
    assert document["dataset"]["examples"] == 14
    assert len(document["cases"]) == 14
    assert all(case["status"] in {"unchanged", "still_wrong"} for case in document["cases"])
    assert not document["representative"]["fixed"]
    assert not document["representative"]["regressed"]


def test_diagnosis_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _run(first)
    _run(second)
    assert first.read_bytes() == second.read_bytes()


def test_ajimee_is_rejected(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/diagnose_cases.py", "--suite", "ajimee",
         "--baseline-weights", "x", "--learned-weights", "x", "--output", str(tmp_path / "out.json")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "sentence-level" in result.stderr
