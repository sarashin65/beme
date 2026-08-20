"""Black-box tests for ``python -m beme convert``."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def write_dictionary(tmp_path: Path) -> Path:
    path = tmp_path / "dictionary.tsv"
    path.write_text(
        "reading\tsurface\tpos\tcost\n"
        "かみ\t神\tnoun\t0\n"
        "かみ\t紙\tnoun\t1\n",
        encoding="utf-8",
    )
    return path


def run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root / "src"), env.get("PYTHONPATH", "")) if part
    )
    return subprocess.run(
        [sys.executable, "-m", "beme", *args],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_text_cli_reports_candidates_costs_and_cascade_signal(tmp_path: Path) -> None:
    process = run_cli(
        tmp_path,
        "convert",
        "かみ",
        "--dictionary",
        str(write_dictionary(tmp_path)),
        "--top-k",
        "2",
        "--cascade-threshold",
        "2.0",
    )

    assert process.returncode == 0, process.stderr
    assert "神" in process.stdout
    assert "紙" in process.stdout
    assert "cost" in process.stdout.lower()
    assert "confidence" in process.stdout.lower()
    assert "rerank" in process.stdout.lower()


def test_json_cli_is_machine_readable_and_honors_top_k(tmp_path: Path) -> None:
    process = run_cli(
        tmp_path,
        "convert",
        "かみ",
        "--dictionary",
        str(write_dictionary(tmp_path)),
        "--top-k",
        "1",
        "--json",
    )

    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    assert payload["reading"] == "かみ"
    assert len(payload["candidates"]) == 1
    assert {"surface", "cost"} <= set(payload["candidates"][0])
    assert "confidence" in payload
    assert "cascade_threshold" in payload
    assert isinstance(payload["needs_rerank"], bool)


def test_invalid_cli_input_is_an_argparse_error_without_traceback(tmp_path: Path) -> None:
    process = run_cli(
        tmp_path,
        "convert",
        "カミ",
        "--dictionary",
        str(write_dictionary(tmp_path)),
    )

    assert process.returncode != 0
    assert "error" in process.stderr.lower()
    assert "traceback" not in process.stderr.lower()
