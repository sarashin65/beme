"""Unit tests for the ablation harness without running training."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


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
