"""Tests for the oracle recall measurement CLI primitives."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(output: Path) -> dict[str, object]:
    subprocess.run(
        [sys.executable, "scripts/oracle_recall.py", "--suite", "handmade",
         "--dictionary", "data/dictionary_v2.tsv", "--weights", "data/weights/decomposed_v1.json",
         "--output", str(output), "--json"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_handmade_measurement_has_separate_metrics(tmp_path: Path) -> None:
    document = _run(tmp_path / "one.json")
    assert document["dataset"]["examples"] == 14
    assert document["beam_width"] == 64
    assert set(document["path_recall_at_k"]) == {"1", "5", "10", "20", "64"}
    assert document["unique_surface_count"]["max"] <= 64


def test_output_is_byte_identical(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _run(first)
    _run(second)
    assert first.read_bytes() == second.read_bytes()


def test_ajimee_is_rejected_with_reason(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/oracle_recall.py", "--suite", "ajimee", "--weights", "x",
         "--output", str(tmp_path / "out.json")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "sentence-level" in result.stderr
