"""Behavioral tests for D1 data construction and conditional ranking metrics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_d1_data = load_script("build_d1_data")
evaluate_d1 = load_script("evaluate_d1")


def make_group(group_id: str, positive: str, scores: dict[str, tuple[float, float]]) -> dict[str, object]:
    candidates = list(scores)
    return {
        "id": group_id,
        "article_id": f"article-{group_id}",
        "left_context": "左文脈",
        "reading": "どうおん",
        "right_context": "右文脈",
        "positive": positive,
        "candidates": candidates,
        "viterbi_costs": [float(index) for index in range(len(candidates))],
        "frequency_bucket": {"positive": "mid", "hard_negative": "high"},
        "span_tokens": 1,
        "contrast_pair_id": "pair-1",
        "_scores": scores,
    }


class FakeScorer:
    def score(self, left: str, reading: str, right: str, candidates: list[str]) -> list[object]:
        group = self.groups.pop(0)
        assert reading == group["reading"]
        return [
            evaluate_d1.CandidateScore(candidate, *group["_scores"][candidate], 1)
            for candidate in candidates
        ]

    def __init__(self, groups: list[dict[str, object]]) -> None:
        self.groups = groups.copy()


def test_article_split_is_stable_and_seeded() -> None:
    first = [build_d1_data.article_split(str(index), 7) for index in range(100)]
    second = [build_d1_data.article_split(str(index), 7) for index in range(100)]
    changed = [build_d1_data.article_split(str(index), 8) for index in range(100)]

    assert first == second
    assert first != changed
    assert {"train", "dev", "test"}.issubset(first)


def test_balanced_selection_caps_repeated_readings() -> None:
    groups = [
        {"id": f"a-{index}", "reading": "a", "positive": str(index)} for index in range(5)
    ] + [
        {"id": f"b-{index}", "reading": "b", "positive": str(index)} for index in range(5)
    ]

    selected = build_d1_data.balanced_select(groups, target=6, per_reading=3, seed=1)

    assert len(selected) == 6
    assert sum(group["reading"] == "a" for group in selected) == 3
    assert sum(group["reading"] == "b" for group in selected) == 3


def test_contrast_pairs_require_different_surfaces() -> None:
    groups = [
        {"id": "1", "reading": "こうしょう", "positive": "考証", "contrast_pair_id": None},
        {"id": "2", "reading": "こうしょう", "positive": "考証", "contrast_pair_id": None},
        {"id": "3", "reading": "こうしょう", "positive": "交渉", "contrast_pair_id": None},
    ]

    assert build_d1_data.add_contrast_pairs(groups) == 1
    paired = [group for group in groups if group["contrast_pair_id"]]
    assert len(paired) == 2
    assert {group["positive"] for group in paired} == {"考証", "交渉"}


def test_conditional_metrics_separate_recovery_and_regression() -> None:
    groups = [
        make_group("one", "正解一", {"誤答一": (-2.0, -2.0), "正解一": (-1.0, -1.0)}),
        make_group("two", "正解二", {"正解二": (-1.0, -1.0), "誤答二": (-2.0, -2.0)}),
    ]
    scorer = FakeScorer(groups)

    report = evaluate_d1.evaluate_groups(groups, scorer, "left+right", None)

    assert report["viterbi"]["conditional_top1"] == 0.5
    assert report["sum"]["conditional_top1"] == 1.0
    assert report["sum"]["recoverable_error_recovery"] == 1.0
    assert report["sum"]["baseline_correct_regression"] == 0.0
    assert report["sum"]["contrast_pair_count"] == 1
    assert report["sum"]["contrast_pair_both_correct"] == 1.0
