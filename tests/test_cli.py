"""Black-box tests for ``python -m beme convert``."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

import beme.cli as beme_cli
from beme.reranker import CrossEncoderRerankerUnavailable


def write_dictionary(tmp_path: Path) -> Path:
    path = tmp_path / "dictionary.tsv"
    path.write_text(
        "reading\tsurface\tpos\tcost\n"
        "かみ\t神\tnoun\t0\n"
        "かみ\t紙\tnoun\t1\n"
        "こうしょう\t交渉\tnoun\t0\n"
        "こうしょう\t考証\tnoun\t0.4\n",
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


def test_cli_accepts_decomposed_weights() -> None:
    root = Path(__file__).resolve().parents[1]
    process = run_cli(
        root,
        "convert",
        "きしゃ",
        "--weights",
        str(root / "data/weights/decomposed_v1.json"),
        "--top-k",
        "1",
        "--json",
    )

    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["reading"] == "きしゃ"


def test_cli_context_options_rerank_and_are_reported_in_json(tmp_path: Path) -> None:
    dictionary = str(write_dictionary(tmp_path))
    history = run_cli(
        tmp_path,
        "convert",
        "こうしょう",
        "--dictionary",
        dictionary,
        "--top-k",
        "1",
        "--preceding-context",
        "歴史資料を調べる",
        "--json",
    )
    business = run_cli(
        tmp_path,
        "convert",
        "こうしょう",
        "--dictionary",
        dictionary,
        "--top-k",
        "1",
        "--following-context",
        "について商談する",
        "--json",
    )

    assert history.returncode == 0, history.stderr
    history_payload = json.loads(history.stdout)
    assert history_payload["candidates"][0]["surface"] == "考証"
    assert history_payload["context"] == {
        "preceding": "歴史資料を調べる",
        "following": "",
    }
    assert history_payload["reranked"] is True

    assert business.returncode == 0, business.stderr
    business_payload = json.loads(business.stdout)
    assert business_payload["candidates"][0]["surface"] == "交渉"
    assert business_payload["context"]["following"] == "について商談する"
    assert business_payload["reranked"] is True


def test_cli_rejects_non_object_weights_json_without_traceback(tmp_path: Path) -> None:
    """Regression for issue #9 Phase 1 review L1.

    The CLI used to pre-read --weights with a bare ``json.load(stream).get(
    "feature_spec", ...)`` to figure out which extractor to build. A
    top-level JSON array (or any non-dict) has no ``.get``, so that call
    raised an uncaught AttributeError -- an unhandled traceback instead of a
    clean argparse error -- even though Weights.load() itself already
    rejects non-dict documents correctly.
    """
    weights_path = tmp_path / "weights.json"
    weights_path.write_text("[]", encoding="utf-8")
    process = run_cli(
        tmp_path,
        "convert",
        "かみ",
        "--dictionary",
        str(write_dictionary(tmp_path)),
        "--weights",
        str(weights_path),
    )

    assert process.returncode != 0
    assert "error" in process.stderr.lower()
    assert "traceback" not in process.stderr.lower()


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


def test_cli_llm_load_failure_falls_back_to_mock_without_traceback(tmp_path: Path) -> None:
    process = run_cli(
        tmp_path,
        "convert",
        "こうしょう",
        "--dictionary",
        str(write_dictionary(tmp_path)),
        "--reranker",
        "llm",
        "--model-path",
        str(tmp_path / "missing.gguf"),
        "--preceding-context",
        "歴史資料を調べる",
        "--json",
    )

    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["candidates"][0]["surface"] == "考証"
    assert "falling back to mock reranker" in process.stderr
    assert "traceback" not in process.stderr.lower()


def test_cli_cross_encoder_load_failure_falls_back_without_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unavailable(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise CrossEncoderRerankerUnavailable("model unavailable in test")

    monkeypatch.setattr(beme_cli, "CrossEncoderReranker", unavailable)
    return_code = beme_cli.main(
        [
            "convert",
            "こうしょう",
            "--dictionary",
            str(write_dictionary(tmp_path)),
            "--reranker",
            "cross-encoder",
            "--preceding-context",
            "歴史資料を調べる",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert return_code == 0
    assert json.loads(captured.out)["candidates"][0]["surface"] == "考証"
    assert "falling back to mock reranker" in captured.err
