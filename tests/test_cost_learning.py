"""Tests for the cost-learning helpers in ``scripts/apply_cost_learning.py``.

These only exercise ``apply_costs`` and ``select_vocabulary``, which take
pre-computed frequency counters as input. ``collect_frequencies`` requires
fugashi/ipadic and a corpus file and is intentionally left untested here.
"""

from __future__ import annotations

import importlib.util
import math
from collections import Counter
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "apply_cost_learning", REPOSITORY_ROOT / "scripts" / "apply_cost_learning.py"
)
assert _SPEC is not None and _SPEC.loader is not None
apply_cost_learning = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(apply_cost_learning)

apply_costs = apply_cost_learning.apply_costs
select_vocabulary = apply_cost_learning.select_vocabulary


def write_dictionary(tmp_path: Path, rows: list[str], name: str = "dictionary.tsv") -> Path:
    path = tmp_path / name
    path.write_text(
        "reading\tsurface\tpos\tcost\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_apply_costs_subtracts_alpha_log10_frequency(tmp_path: Path) -> None:
    dictionary = write_dictionary(tmp_path, ["する\tする\tverb\t2.000"])
    output = tmp_path / "learned.tsv"
    frequencies: Counter[tuple[str, str, str]] = Counter({("する", "する", "verb"): 100})

    matched, changed = apply_costs(dictionary, output, frequencies, top_n=10, alpha=0.5, min_cost=0.1)

    assert matched == 1
    assert changed == 1
    expected_cost = 2.000 - 0.5 * math.log10(100)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[1] == f"する\tする\tverb\t{expected_cost:.3f}"


def test_apply_costs_clips_at_min_cost(tmp_path: Path) -> None:
    dictionary = write_dictionary(tmp_path, ["ある\tある\tverb\t1.000"])
    output = tmp_path / "learned.tsv"
    # log10(10_000) == 4, so alpha=1.0 would push the cost to -3.0 without clipping.
    frequencies: Counter[tuple[str, str, str]] = Counter({("ある", "ある", "verb"): 10_000})

    matched, changed = apply_costs(dictionary, output, frequencies, top_n=10, alpha=1.0, min_cost=0.5)

    assert matched == 1
    assert changed == 1
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "ある\tある\tverb\t0.500"


def test_apply_costs_leaves_row_untouched_when_cost_would_not_decrease(tmp_path: Path) -> None:
    # log10(1) == 0, so the subtracted amount is zero regardless of alpha:
    # the row is matched but must not be rewritten or counted as changed.
    dictionary = write_dictionary(tmp_path, ["ねこ\t猫\tnoun\t2.000"])
    output = tmp_path / "learned.tsv"
    frequencies: Counter[tuple[str, str, str]] = Counter({("ねこ", "猫", "noun"): 1})

    matched, changed = apply_costs(dictionary, output, frequencies, top_n=10, alpha=0.5, min_cost=0.1)

    assert matched == 1
    assert changed == 0
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "ねこ\t猫\tnoun\t2.000"


def test_apply_costs_ignores_frequencies_not_present_in_dictionary(tmp_path: Path) -> None:
    dictionary = write_dictionary(tmp_path, ["いぬ\t犬\tnoun\t1.500"])
    output = tmp_path / "learned.tsv"
    # None of these tuples match the (reading, surface, pos) row above.
    frequencies: Counter[tuple[str, str, str]] = Counter(
        {("いぬ", "犬", "verb"): 500, ("ねこ", "猫", "noun"): 500}
    )

    matched, changed = apply_costs(dictionary, output, frequencies, top_n=10, alpha=0.5, min_cost=0.1)

    assert matched == 0
    assert changed == 0
    assert output.read_text(encoding="utf-8") == dictionary.read_text(encoding="utf-8")


def test_apply_costs_rejects_invalid_header(tmp_path: Path) -> None:
    dictionary = tmp_path / "bad-header.tsv"
    dictionary.write_text("surface\treading\tpos\tcost\nかみ\t神\tnoun\t1.0\n", encoding="utf-8")
    output = tmp_path / "learned.tsv"

    with pytest.raises(ValueError):
        apply_costs(dictionary, output, Counter(), top_n=10, alpha=0.1, min_cost=0.1)


def test_apply_costs_rejects_rows_with_wrong_field_count(tmp_path: Path) -> None:
    dictionary = tmp_path / "malformed.tsv"
    dictionary.write_text(
        "reading\tsurface\tpos\tcost\nかみ\t神\tnoun\n",
        encoding="utf-8",
    )
    output = tmp_path / "learned.tsv"

    with pytest.raises(ValueError):
        apply_costs(dictionary, output, Counter(), top_n=10, alpha=0.1, min_cost=0.1)


def test_apply_costs_does_not_modify_input_when_output_path_differs(tmp_path: Path) -> None:
    dictionary = write_dictionary(tmp_path, ["する\tする\tverb\t2.000"])
    output = tmp_path / "learned.tsv"
    original_bytes = dictionary.read_bytes()
    frequencies: Counter[tuple[str, str, str]] = Counter({("する", "する", "verb"): 100})

    apply_costs(dictionary, output, frequencies, top_n=10, alpha=0.5, min_cost=0.1)

    assert dictionary.read_bytes() == original_bytes
    assert output.exists()
    assert output != dictionary


def test_apply_costs_can_overwrite_the_input_path(tmp_path: Path) -> None:
    dictionary = write_dictionary(tmp_path, ["する\tする\tverb\t2.000"])
    frequencies: Counter[tuple[str, str, str]] = Counter({("する", "する", "verb"): 100})

    matched, changed = apply_costs(dictionary, dictionary, frequencies, top_n=10, alpha=0.5, min_cost=0.1)

    assert matched == 1
    assert changed == 1
    expected_cost = 2.000 - 0.5 * math.log10(100)
    lines = dictionary.read_text(encoding="utf-8").splitlines()
    assert lines[1] == f"する\tする\tverb\t{expected_cost:.3f}"


def test_select_vocabulary_truncates_to_top_n() -> None:
    frequencies: Counter[tuple[str, str, str]] = Counter(
        {
            ("あ", "亜", "noun"): 5,
            ("い", "衣", "noun"): 50,
            ("う", "宇", "noun"): 20,
            ("え", "絵", "noun"): 1,
            ("お", "尾", "noun"): 10,
        }
    )

    selected = select_vocabulary(frequencies, top_n=3)

    assert len(selected) == 3
    assert set(selected) == {("い", "衣", "noun"), ("う", "宇", "noun"), ("お", "尾", "noun")}


def test_select_vocabulary_breaks_ties_deterministically_by_tuple_order() -> None:
    frequencies: Counter[tuple[str, str, str]] = Counter(
        {
            ("う", "宇", "noun"): 10,
            ("あ", "亜", "noun"): 10,
            ("い", "衣", "noun"): 10,
        }
    )

    selected = select_vocabulary(frequencies, top_n=10)

    # Equal frequencies must be ordered by the (reading, surface, pos) tuple.
    assert list(selected.keys()) == [
        ("あ", "亜", "noun"),
        ("い", "衣", "noun"),
        ("う", "宇", "noun"),
    ]


def test_select_vocabulary_rejects_non_positive_top_n() -> None:
    with pytest.raises(ValueError):
        select_vocabulary(Counter({("あ", "亜", "noun"): 1}), top_n=0)
