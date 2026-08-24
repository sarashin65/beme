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


def test_mcnemar_is_imported_from_rerank_stats_not_redefined() -> None:
    """Regression for issue #9 Phase 4 review item 6.

    Task 3 (issue #9 Phase 4) introduced src/beme/rerank_stats.py precisely so
    the statistical decision logic would live in one tested place; this
    ablation harness had its own byte-for-byte copy of `mcnemar` that never
    got wired to it. Assert identity, not just matching output, so a future
    edit to one cannot silently drift from the other again.
    """
    from beme.rerank_stats import mcnemar as shared_mcnemar

    value = module()
    assert value.mcnemar is shared_mcnemar


@pytest.mark.parametrize(
    "b, c, expected, method",
    [
        # Exact: two-sided binomial, p = 2 * P(X <= min(b, c)) capped at 1.
        (5, 0, 2 * (1 / 32), "exact_binomial_two_sided"),          # 2 * 1/2^5
        (0, 4, 2 * (1 / 16), "exact_binomial_two_sided"),          # 2 * 1/2^4
        (2, 1, 1.0, "exact_binomial_two_sided"),                   # 2 * (1+3)/8, capped
        (0, 0, 1.0, "exact_binomial_two_sided"),                   # no discordant pairs
        # Continuity-corrected chi-square on 1 df, p = erfc(sqrt(chi2 / 2)).
        (20, 10, 0.10034824646229071, "continuity_corrected_chi_squared"),  # chi2 = 9^2/30
        (30, 10, 0.0026631192591385546, "continuity_corrected_chi_squared"),  # chi2 = 19^2/40
    ],
)
def test_mcnemar_matches_hand_computed_values(b: int, c: int, expected: float, method: str) -> None:
    value = module()
    table = value.mcnemar(b, c)
    assert table["method"] == method
    assert table["p_value"] == pytest.approx(expected, rel=1e-12)
    assert (table["b"], table["c"], table["discordant"]) == (b, c, b + c)


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


def _significant_cases(baseline_correct: int, learned_correct: int, moved_wrong: int, moved_right: int) -> list[dict]:
    """A0-learned-correct block of `learned_correct`, then move cases each way."""
    cases = _cases(baseline_correct, learned_correct)
    for index in range(moved_wrong):
        cases[index]["learned"]["correct"] = False
    for index in range(learned_correct, learned_correct + moved_right):
        cases[index]["learned"]["correct"] = True
    return cases


def test_one_significant_seed_out_of_three_is_not_a_label(tmp_path: Path) -> None:
    """A difference only one seed can see is not one the condition reproduces."""
    value = module()
    for seed in value.SEEDS:
        _artifacts(tmp_path, "A0", seed, _cases(50, 80), 20)
    # seed 9 moves 30 cases one way (significant); the other two barely move.
    _artifacts(tmp_path, "A1", value.SEEDS[0], _significant_cases(50, 80, 30, 0), 10)
    for seed in value.SEEDS[1:]:
        _artifacts(tmp_path, "A1", seed, _significant_cases(50, 80, 1, 0), 10)
    summary = value.summarize_only(tmp_path)
    decision = summary["conditions"]["A1"]["decision"]
    assert [round(item["table"]["p_value"], 3) for item in summary["conditions"]["A1"]["mcnemar"]][0] < 0.05
    assert decision["max_p_value"] >= 0.05
    assert decision["seeds_compared"] == 3
    assert decision["label"] == "indeterminate"
    assert summary["conditions"]["A1"]["decision_label"] == "indeterminate"


def test_all_seeds_significant_and_the_median_agrees_gives_a_label(tmp_path: Path) -> None:
    value = module()
    for seed in value.SEEDS:
        _artifacts(tmp_path, "A0", seed, _cases(50, 80), 20)
        _artifacts(tmp_path, "A1", seed, _significant_cases(50, 80, 30, 0), 50)
    summary = value.summarize_only(tmp_path)
    decision = summary["conditions"]["A1"]["decision"]
    assert decision["max_p_value"] < 0.05
    assert decision["label"] == "significant_worse"


def test_a_label_may_rest_on_fewer_seeds_but_records_how_many(tmp_path: Path) -> None:
    """A3 lost one seed's artifact; the label must say what it rests on."""
    value = module()
    for seed in value.SEEDS:
        _artifacts(tmp_path, "A0", seed, _cases(50, 80), 20)
    for seed in value.SEEDS:
        cases = _significant_cases(50, 80, 30, 0)
        _artifacts(tmp_path, "A1", seed, cases, 50)
    (tmp_path / "A1" / f"seed{value.SEEDS[2]}" / "diagnose_d1.json").unlink()
    summary = value.summarize_only(tmp_path)
    decision = summary["conditions"]["A1"]["decision"]
    assert decision["seeds_compared"] == 2
    assert decision["label"] == "significant_worse"
