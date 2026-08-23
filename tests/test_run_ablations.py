"""Unit tests for the ablation harness without running training."""

from __future__ import annotations

import importlib.util
import json
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


def _artifacts(root: Path, condition: str, seed: int, cases: list[dict], errors: int) -> None:
    directory = root / condition / f"seed{seed}"
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(
        json.dumps({"selected_model": {"test": {"end_to_end_errors": errors}}}), encoding="utf-8"
    )
    (directory / "evaluate.json").write_text("{}", encoding="utf-8")
    (directory / "weights.json").write_text(
        json.dumps({"weights": {"lex.ipadic_cost_mapped": 1.0}}), encoding="utf-8"
    )
    (directory / "diagnose_d1.json").write_text(
        json.dumps({"dataset": {"examples": len(cases)}, "cases": cases}), encoding="utf-8"
    )


def _cases(baseline_correct: int, learned_correct: int, total: int = 100) -> list[dict]:
    return [
        {
            "id": str(index),
            "baseline": {"correct": index < baseline_correct},
            "learned": {"correct": index < learned_correct},
        }
        for index in range(total)
    ]


def test_mcnemar_compares_the_learned_models_not_the_untrained_baseline(tmp_path: Path) -> None:
    """A condition equal to A0 must be indistinguishable from A0.

    The table must not compare against the untrained baseline: that document
    is identical for every condition, so it would report a large significant
    difference for a condition that never differs from A0 at all.
    """
    value = module()
    for seed in value.SEEDS:
        _artifacts(tmp_path, "A0", seed, _cases(50, 80), 20)
        _artifacts(tmp_path, "A1", seed, _cases(50, 80), 20)
    summary = value.summarize_only(tmp_path)
    table = summary["conditions"]["A1"]["mcnemar"][0]["table"]
    assert (table["b"], table["c"], table["p_value"]) == (0, 0, 1.0)
    assert summary["conditions"]["A1"]["decision_label"] == "indeterminate"


def test_mcnemar_pairs_each_condition_seed_with_the_same_a0_seed(tmp_path: Path) -> None:
    value = module()
    for index, seed in enumerate(value.SEEDS):
        _artifacts(tmp_path, "A0", seed, _cases(50, 80), 20)
        # Each seed differs from A0 by a different number of cases.
        _artifacts(tmp_path, "A1", seed, _cases(50, 80 - index), 20 + index)
    summary = value.summarize_only(tmp_path)
    tables = {item["seed"]: item["table"] for item in summary["conditions"]["A1"]["mcnemar"]}
    assert [tables[seed]["b"] for seed in value.SEEDS] == [0, 1, 2]
    assert all(table["c"] == 0 for table in tables.values())


def test_equal_medians_are_indeterminate_even_when_the_table_is_significant(tmp_path: Path) -> None:
    value = module()
    for seed in value.SEEDS:
        _artifacts(tmp_path, "A0", seed, _cases(50, 80), 20)
        # A lopsided 30/10 table, but the recorded error count matches A0.
        moved = _cases(50, 80)
        for index in range(30):
            moved[index]["learned"]["correct"] = False
        for index in range(80, 90):
            moved[index]["learned"]["correct"] = True
        _artifacts(tmp_path, "A1", seed, moved, 20)
    summary = value.summarize_only(tmp_path)
    table = summary["conditions"]["A1"]["mcnemar"][0]["table"]
    assert (table["b"], table["c"]) == (30, 10)
    assert table["p_value"] < 0.05
    assert summary["conditions"]["A1"]["decision_label"] == "indeterminate"
