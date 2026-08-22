"""Tests for cost-scale measurement helpers and self-consistency."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_quantiles_and_null_confidences() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("measure_cost_scale", ROOT / "scripts/measure_cost_scale.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = module.confidence_summary([1.0, 2.0, 3.0, None], 2.0)
    assert result["null_confidence_count"] == 1
    assert result["p50"] == 2.0


def test_handmade_selfcheck_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    command = [sys.executable, "scripts/measure_cost_scale.py", "--suite", "handmade", "--dictionary", "data/dictionary_v2.tsv", "--baseline-weights", "data/weights/decomposed_v1.json", "--learned-weights", "data/weights/decomposed_v1.json"]
    for output in (first, second):
        subprocess.run([*command, "--output", str(output)], cwd=ROOT, check=True)
    assert first.read_bytes() == second.read_bytes()
    document = json.loads(first.read_text(encoding="utf-8"))
    section = document["confidence_distributions"]["handmade"]
    assert section["scale_ratio"] == 1.0
    assert section["threshold_learned"] == 2.0
