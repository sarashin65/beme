"""Unit tests for the ablation harness without running training."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import pytest


ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location("run_ablations", ROOT / "scripts" / "run_ablations.py")
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def test_condition_table_and_mcnemar_methods() -> None:
    value = module()
    assert len(value.CONDITIONS) == 12
    assert value.mcnemar(2, 1)["method"] == "exact_binomial_two_sided"
    assert value.mcnemar(20, 10)["method"] == "continuity_corrected_chi_squared"


def test_summary_statistics() -> None:
    value = module()
    assert value.summarize_values([3, 1, 2]) == {"median": 2, "min": 1, "max": 3, "range": [1, 3]}


@pytest.mark.parametrize("force, expected", [(False, False), (True, True)])
def test_a0_gate_force_controls_rerun(tmp_path: Path, force: bool, expected: bool) -> None:
    value = module()
    gate = tmp_path / "A0" / "seed9" / "a0_gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_text("{}", encoding="utf-8")
    assert value._should_run_a0(["A0"], [9], force, tmp_path) is expected


def test_pending_runs_always_excludes_a0_seed9() -> None:
    value = module()
    assert value._pending_runs(["A0", "A1"], [9, 17]) == [("A0", 17), ("A1", 9), ("A1", 17)]
