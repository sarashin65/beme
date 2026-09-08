import json
import subprocess
from unittest.mock import Mock

from scripts.e0_llm_prompt import SCHEMA_VERSION
from scripts.e0_llm_runner import call_claude_cli_batch, parse_llm_response, run_annotation


def _annotation() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": "c1",
        "input_reading": "かな。",
        "canonical_reference": "仮名。",
        "tokens": [
            {"surface": "仮名", "reading": "かな", "surface_start": 0, "surface_end": 2,
             "reading_start": 0, "reading_end": 2, "is_literal": False},
            {"surface": "。", "reading": "。", "surface_start": 2, "surface_end": 3,
             "reading_start": 2, "reading_end": 3, "is_literal": True},
        ],
    }


def test_parse_valid_json_and_concatenation() -> None:
    result = parse_llm_response(json.dumps(_annotation(), ensure_ascii=False))
    assert result["ok"]
    assert result["data"]["case_id"] == "c1"


def test_parse_rejects_broken_json_and_bad_offsets() -> None:
    assert parse_llm_response("{broken")["error"] == "invalid_json"
    annotation = _annotation()
    annotation["tokens"][1]["reading_start"] = 99
    result = parse_llm_response(json.dumps(annotation, ensure_ascii=False))
    assert not result["ok"]
    assert result["error"] == "reading_offsets_invalid"


def test_dry_run_returns_mecab_tokens_and_saves_raw(tmp_path) -> None:
    tokens = [
        {"surface": "仮名", "reading": "かな", "surface_start": 0, "surface_end": 2,
         "reading_start": 0, "reading_end": 2, "is_literal": False},
        {"surface": "。", "reading": "。", "surface_start": 2, "surface_end": 3,
         "reading_start": 2, "reading_end": 3, "is_literal": True},
    ]
    result = run_annotation([{"id": "c1", "source_text": "仮名。", "gold_tokens": tokens}], tmp_path, dry_run=True)
    assert len(result) == 1
    assert result[0]["gold_paths"][0]["tokens"] == tokens
    raw_files = list((tmp_path / "raw").glob("*.jsonl"))
    assert len(raw_files) == 1
    raw = json.loads(raw_files[0].read_text(encoding="utf-8").splitlines()[0])
    assert raw["case_id"] == "c1"
    assert raw["raw_response"]
    assert (tmp_path / "annotation_failures.jsonl").read_text(encoding="utf-8") == ""


def test_claude_cli_batch_uses_ssh_and_parses_json(monkeypatch) -> None:
    completed = Mock(stdout=json.dumps([_annotation()], ensure_ascii=False))
    run = Mock(return_value=completed)
    monkeypatch.setattr("scripts.e0_llm_runner.subprocess.run", run)

    result = call_claude_cli_batch([{"id": "c1", "source_text": "仮名。"}])

    assert result[0]["case_id"] == "c1"
    command = run.call_args.args[0]
    assert command == ["ssh", "ws", "claude", "--model", "opus", "-p"]
    prompt = run.call_args.kwargs["input"]
    assert '"case_id":"c1"' in prompt
    assert "JSON 配列" in prompt
    assert run.call_args.kwargs["timeout"] == 120


def test_claude_cli_splits_candidates_into_batches_of_50(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_batch(cases, model="opus", ssh_host="ws"):
        calls.append(cases)
        return [
            {**_annotation(), "case_id": str(case["id"]),
             "canonical_reference": case["source_text"]}
            for case in cases
        ]

    monkeypatch.setattr("scripts.e0_llm_runner.call_claude_cli_batch", fake_batch)
    cases = [{"id": f"c{i}", "source_text": "仮名。"} for i in range(51)]

    result = run_annotation(cases, tmp_path, model="opus", backend="claude-cli")

    assert [len(batch) for batch in calls] == [50, 1]
    assert len(result) == 51
    raw_lines = next((tmp_path / "raw").glob("*.jsonl")).read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 2
    assert len(json.loads(raw_lines[0])["case_ids"]) == 50


def test_claude_cli_retries_one_failed_batch(monkeypatch) -> None:
    completed = Mock(stdout=json.dumps([_annotation()], ensure_ascii=False))
    run = Mock(side_effect=[subprocess.CalledProcessError(1, "ssh"), completed])
    monkeypatch.setattr("scripts.e0_llm_runner.subprocess.run", run)

    result = call_claude_cli_batch([{"id": "c1", "source_text": "仮名。"}])

    assert result[0]["case_id"] == "c1"
    assert run.call_count == 2
