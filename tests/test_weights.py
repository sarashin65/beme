"""Persistence and process-boundary tests for scorer weights."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from beme.features import DECOMPOSED_V1_SPEC, legacy_v1_extractor
from beme.scorer import LinearScorer
from beme.weights import DEFAULT_WEIGHTS_PATH, Weights

DECOMPOSED_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "data" / "weights" / "decomposed_v1.json"


def test_weights_round_trip_is_deterministic(tmp_path: Path) -> None:
    weights = Weights.load(DEFAULT_WEIGHTS_PATH)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    weights.dump(first)
    Weights.load(first).dump(second)
    assert json.loads(first.read_text()) == json.loads(second.read_text())
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize(
    ("field", "value"),
    [("format", "beme.weights/999"), ("feature_spec", "other_v1")],
)
def test_weights_reject_incompatible_metadata(tmp_path: Path, field: str, value: str) -> None:
    document = json.loads(DEFAULT_WEIGHTS_PATH.read_text())
    document[field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        Weights.load(path)


def test_weights_unknown_feature_strict_flag(tmp_path: Path) -> None:
    document = json.loads(DEFAULT_WEIGHTS_PATH.read_text())
    document["weights"]["future.feature"] = 2.0
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown feature"):
        Weights.load(path)
    with pytest.warns(UserWarning, match="unknown feature"):
        loaded = Weights.load(path, strict=False)
    assert loaded["future.feature"] == 2.0


def test_decomposed_script_features_round_trip(tmp_path: Path) -> None:
    """Regression for issue #9 Phase 1 review M1.

    FeatureSpec.decomposed_v1() used to declare the glob string
    "lex.script.*" while FeatureExtractor actually emits
    "lex.script.kanji"/"lex.script.hiragana"/"lex.script.katakana"/
    "lex.script.other". Weights.load() only recognizes literal names, so a
    weights file that a Phase 2 learner wrote after putting weight on any
    script feature could never be loaded back, strict or not. Before the fix
    this raised "unknown feature name(s): lex.script.kanji" for every one of
    them.
    """
    document = json.loads(DECOMPOSED_WEIGHTS_PATH.read_text())
    script_names = [name for name in DECOMPOSED_V1_SPEC.lexical_features if name.startswith("lex.script.")]
    assert script_names == [
        "lex.script.kanji", "lex.script.hiragana", "lex.script.katakana", "lex.script.other",
    ]
    for name in script_names:
        document["weights"][name] = 0.5
    path = tmp_path / "decomposed_with_scripts.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = Weights.load(path, feature_spec="decomposed_v1")

    for name in script_names:
        assert loaded[name] == 0.5


def test_weights_load_rejects_non_object_json_cleanly(tmp_path: Path) -> None:
    """Regression for issue #9 Phase 1 review L1.

    Weights.load() already validated a non-dict JSON document correctly; the
    bug was CLI/evaluate.py pre-reading the file with a bare ``.get()`` call
    to peek "feature_spec" before ever reaching this check, which raised an
    uncaught AttributeError for a top-level JSON array or scalar. This test
    guards Weights.load() itself, which callers are now expected to rely on
    exclusively (feature_spec=None) instead of re-parsing the file.
    """
    path = tmp_path / "not_an_object.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        Weights.load(path, feature_spec=None)


def test_weights_load_feature_spec_none_autodetects_and_still_validates(tmp_path: Path) -> None:
    """Regression for issue #9 Phase 1 review L1.

    feature_spec=None must not just skip validation outright: the loaded
    Weights.feature_spec should reflect what the file actually declares, and
    unknown feature names for that declared spec must still be rejected.
    """
    loaded = Weights.load(DECOMPOSED_WEIGHTS_PATH, feature_spec=None)
    assert loaded.feature_spec == "decomposed_v1"

    document = json.loads(DECOMPOSED_WEIGHTS_PATH.read_text())
    document["weights"]["totally.unknown"] = 1.0
    path = tmp_path / "unknown_decomposed.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown feature"):
        Weights.load(path, feature_spec=None)


def _run_convert(weight_path: Path, dictionary: Path) -> str:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "beme",
            "convert",
            "きしゃ",
            "--dictionary",
            str(dictionary),
            "--weights",
            str(weight_path),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return process.stdout


def test_weights_are_identical_in_a_separate_process(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.tsv"
    dictionary.write_text(
        "reading\tsurface\tpos\tcost\nきしゃ\t記者\tnoun\t0.8\nきしゃ\t汽車\tnoun\t0.9\n",
        encoding="utf-8",
    )
    copied = tmp_path / "copied.json"
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    assert _run_convert(DEFAULT_WEIGHTS_PATH, dictionary) == _run_convert(copied, dictionary)


def test_changing_default_transition_weight_changes_one_best(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.tsv"
    dictionary.write_text(
        "reading\tsurface\tpos\tcost\n"
        "きしゃ\t記者\tnoun\t0.2\n"
        "き\t気\tnoun\t0\n"
        "しゃ\t車\tverb\t0\n",
        encoding="utf-8",
    )
    document = json.loads(DEFAULT_WEIGHTS_PATH.read_text())
    document["weights"]["trans.default"] = 5.0
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(document), encoding="utf-8")
    baseline = json.loads(_run_convert(DEFAULT_WEIGHTS_PATH, dictionary))
    changed = json.loads(_run_convert(altered, dictionary))
    assert baseline["candidates"][0]["surface"] != changed["candidates"][0]["surface"]


def test_loaded_weights_drive_linear_scorer() -> None:
    scorer = LinearScorer(legacy_v1_extractor(), Weights.load(DEFAULT_WEIGHTS_PATH))
    assert scorer.spec_id == "legacy_v1"
