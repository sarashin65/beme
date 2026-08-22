"""Reproducibility and split-separation tests for train_scorer.py."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from beme.weights import Weights


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_scorer.py"


def make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    dictionary = tmp_path / "dictionary.tsv"
    dictionary.write_text(
        "reading\tsurface\tpos\tcost\n"
        "か\tA\tnoun\t1\n"
        "か\tB\tnoun\t3\n",
        encoding="utf-8",
    )
    weights = tmp_path / "weights.json"
    weights.write_text(
        json.dumps({
            "format": "beme.weights/1", "feature_spec": "legacy_v1",
            "default_weight": 0.0, "weights": {"lex.base_cost": 1.0, "trans.bos": 0.0},
        }, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text('{"reading":"か","target_surface":"B"}\n', encoding="utf-8")
    dev.write_text('{"reading":"か","target_surface":"B"}\n', encoding="utf-8")
    test.write_text('{"reading":"か","target_surface":"A"}\n', encoding="utf-8")
    return dictionary, weights, train, dev, test


def run_cli(tmp_path: Path, seed: int = 4) -> tuple[bytes, bytes]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dictionary, weights, train, dev, test = make_inputs(tmp_path)
    output_weights = tmp_path / "out" / "weights.json"
    output_report = tmp_path / "out" / "report.json"
    output_weights.parent.mkdir()
    subprocess.run(
        [
            sys.executable, str(SCRIPT), "--train", str(train), "--dev", str(dev), "--test", str(test),
            "--dictionary", str(dictionary), "--initial-weights", str(weights),
            "--epochs", "2", "--learning-rate", "1", "--seed", str(seed),
            "--output-weights", str(output_weights), "--output-report", str(output_report),
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return output_weights.read_bytes(), output_report.read_bytes()


def test_cli_artifacts_are_byte_identical_and_weights_load(tmp_path: Path) -> None:
    weights_a, report_a = run_cli(tmp_path / "a")
    weights_b, report_b = run_cli(tmp_path / "b")
    assert weights_a == weights_b
    assert report_a == report_b
    loaded = Weights.load(tmp_path / "a" / "out" / "weights.json", feature_spec=None)
    assert loaded.metadata["averaged"] is True


def test_report_contains_provenance_baseline_selection_and_test(tmp_path: Path) -> None:
    run_cli(tmp_path)
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["manifest"]["splits"]["train"]["examples"] == 1
    assert len(report["manifest"]["splits"]["train"]["sha256"]) == 64
    assert "dev" in report["baseline"] and "test" in report["baseline"]
    assert report["selected_epoch"] == 1
    assert "test" in report["selected_model"]
    assert report["cli"]["seed"] == 4
    serialized = json.dumps(report)
    assert str(tmp_path) not in serialized
    assert "/tmp/" not in serialized


def test_invalid_arguments_and_missing_inputs_fail(tmp_path: Path) -> None:
    dictionary, weights, train, dev, test = make_inputs(tmp_path)
    common = ["--train", str(train), "--dev", str(dev), "--test", str(test), "--dictionary", str(dictionary), "--initial-weights", str(weights), "--seed", "1", "--output-weights", str(tmp_path / "out" / "w.json"), "--output-report", str(tmp_path / "out" / "r.json")]
    for option, value in (("--epochs", "0"), ("--learning-rate", "0")):
        process = subprocess.run([sys.executable, str(SCRIPT), *common, option, value], cwd=ROOT, capture_output=True, text=True)
        assert process.returncode != 0
    process = subprocess.run([sys.executable, str(SCRIPT), *common, "--epochs", "1", "--learning-rate", "1"], cwd=ROOT, capture_output=True, text=True)
    assert process.returncode != 0


def test_test_contents_do_not_select_the_epoch_and_seed_is_recorded(tmp_path: Path) -> None:
    dictionary, weights, train, dev, test = make_inputs(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    common = [
        "--train", str(train), "--dev", str(dev), "--test", str(test),
        "--dictionary", str(dictionary), "--initial-weights", str(weights),
        "--epochs", "2", "--learning-rate", "1", "--output-weights",
        str(output / "weights.json"), "--output-report", str(output / "report.json"),
    ]
    first = subprocess.run([sys.executable, str(SCRIPT), *common, "--seed", "4"], cwd=ROOT, check=True)
    first_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    test.write_text('{"reading":"か","target_surface":"B"}\n', encoding="utf-8")
    second = subprocess.run([sys.executable, str(SCRIPT), *common, "--seed", "5"], cwd=ROOT, check=True)
    second_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert first.returncode == second.returncode == 0
    assert first_report["selected_epoch"] == second_report["selected_epoch"]
    assert first_report["cli"]["seed"] == 4
    assert second_report["cli"]["seed"] == 5


def test_feature_spec_mismatch_fails(tmp_path: Path) -> None:
    dictionary, weights, train, dev, test = make_inputs(tmp_path)
    document = json.loads(weights.read_text(encoding="utf-8"))
    document["feature_spec"] = "unsupported_v1"
    weights.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    process = subprocess.run(
        [sys.executable, str(SCRIPT), "--train", str(train), "--dev", str(dev), "--test", str(test),
         "--dictionary", str(dictionary), "--initial-weights", str(weights), "--epochs", "1",
         "--learning-rate", "1", "--seed", "1", "--output-weights", str(output / "w.json"),
         "--output-report", str(output / "r.json")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert process.returncode != 0
