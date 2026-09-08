"""Run and validate E0 LLM annotations, with an API-free dry-run mode."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Works both as ``python -m`` and as a direct script.
    from scripts.e0_llm_prompt import SCHEMA_VERSION, build_llm_messages
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI smoke command
    from e0_llm_prompt import SCHEMA_VERSION, build_llm_messages

SEED = 20260908
TIMEOUT_SECONDS = 120
_JSON_KEYS = {"schema_version", "case_id", "input_reading", "canonical_reference", "tokens",
              "alternative_paths", "corrections_from_mecab", "risk_flags", "needs_review", "review_reasons"}
_HIRAGANA = re.compile(r"^[\u3041-\u3096ー]*$")


def _text_from_response(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        if isinstance(response.get("text"), str):
            return response["text"]
        content = response.get("content")
        if isinstance(content, list):
            return "".join(item.get("text", "") for item in content if isinstance(item, Mapping))
    content = getattr(response, "content", None)
    if content:
        return "".join(getattr(item, "text", "") for item in content)
    return str(response)


def call_llm(messages: Mapping[str, Any] | Sequence[Mapping[str, Any]], model: str = "claude-opus-5-20250514") -> dict[str, Any]:
    """Call Anthropic once, retrying once; return text plus immutable raw text."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required unless --dry-run is used")
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise RuntimeError("anthropic package is required for live annotation") from exc
    if isinstance(messages, Mapping):
        system = str(messages.get("system", ""))
        api_messages = list(messages.get("messages", []))
    else:
        system = ""
        api_messages = list(messages)
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=TIMEOUT_SECONDS)
            response = client.messages.create(
                model=model, system=system, messages=api_messages,
                max_tokens=4096, temperature=0,
            )
            text = _text_from_response(response)
            return {"text": text, "raw_response": text, "model": model,
                    "attempt": attempt + 1, "response_id": getattr(response, "id", "")}
        except Exception as exc:  # API/network errors are retryable once.
            last_error = exc
            if attempt == 0:
                time.sleep(0.2)
    raise RuntimeError(f"LLM request failed after retry: {last_error}") from last_error


def _claude_cli_prompt(cases: Sequence[Mapping[str, Any]]) -> str:
    payloads = []
    for case in cases:
        mecab = list(case.get("mecab_tokens", case.get("gold_tokens", [])))
        built = build_llm_messages(case, mecab, case.get("d1_anchors", []))
        payloads.append(built["payload"])
    return (
        f"{build_llm_messages({}, [], [])['system']}\n\n"
        "以下の各 case の読み・境界を注釈し、入力順を保った JSON 配列で返してください。"
        "説明、Markdown、コードフェンスは返さないでください。各要素は E0 schema の"
        " annotation object（schema_version, case_id, input_reading, canonical_reference, "
        "tokens と任意の review fields）にしてください。\n"
        f"cases={json.dumps(payloads, ensure_ascii=False, separators=(',', ':'))}"
    )


def _call_claude_cli_text(cases: Sequence[Mapping[str, Any]], model: str, ssh_host: str) -> str:
    prompt = _claude_cli_prompt(cases)
    command = ["ssh", ssh_host, "claude", "--model", model, "-p"]
    try:
        completed = subprocess.run(
            command, input=prompt, text=True, capture_output=True,
            check=True, timeout=TIMEOUT_SECONDS,
        )
        return completed.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"claude CLI request failed: {exc}") from exc


def call_claude_cli_batch(
    cases: list[dict], model: str = "opus", ssh_host: str = "ws"
) -> list[dict]:
    """Annotate a batch through the Claude CLI running on the ws host.

    The CLI is deliberately invoked through ``ssh`` and receives the prompt on
    stdin.  A failed subprocess is retried once; malformed JSON is treated as
    a failed batch and is also retried once.
    """
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            text = _call_claude_cli_text(cases, model, ssh_host).strip()
            data = json.loads(text)
            if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
                raise ValueError("claude CLI response must be a JSON array of objects")
            return data
        except (json.JSONDecodeError, TypeError, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.2)
    raise RuntimeError(f"claude CLI batch failed after retry: {last_error}") from last_error


def parse_llm_response(raw_response: Any) -> dict[str, Any]:
    """Parse and fail-closed validate an annotation response."""
    text = _text_from_response(raw_response).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return {"ok": False, "error": "invalid_json", "detail": str(exc), "raw_response": text}
    if not isinstance(data, dict):
        return {"ok": False, "error": "root_not_object", "raw_response": text}
    unknown = sorted(set(data) - _JSON_KEYS)
    required = {"schema_version", "case_id", "input_reading", "canonical_reference", "tokens"}
    missing = sorted(required - set(data))
    if unknown or missing or data.get("schema_version") != SCHEMA_VERSION:
        return {"ok": False, "error": "schema_error", "unknown": unknown, "missing": missing,
                "raw_response": text}
    if not isinstance(data["tokens"], list) or not data["tokens"]:
        return {"ok": False, "error": "tokens_invalid", "raw_response": text}
    reading_cursor = surface_cursor = 0
    for index, token in enumerate(data["tokens"]):
        if not isinstance(token, dict) or set(token) != {"surface", "reading", "surface_start", "surface_end", "reading_start", "reading_end", "is_literal"}:
            return {"ok": False, "error": "token_schema_error", "index": index, "raw_response": text}
        surface, reading = token["surface"], token["reading"]
        if not isinstance(surface, str) or not isinstance(reading, str) or not surface:
            return {"ok": False, "error": "token_value_error", "index": index, "raw_response": text}
        if token["surface_start"] != surface_cursor or token["surface_end"] != surface_cursor + len(surface):
            return {"ok": False, "error": "surface_offsets_invalid", "index": index, "raw_response": text}
        if token["reading_start"] != reading_cursor or token["reading_end"] != reading_cursor + len(reading):
            return {"ok": False, "error": "reading_offsets_invalid", "index": index, "raw_response": text}
        if not isinstance(token["is_literal"], bool):
            return {"ok": False, "error": "literal_invalid", "index": index, "raw_response": text}
        if not token["is_literal"] and not _HIRAGANA.fullmatch(reading):
            return {"ok": False, "error": "reading_not_hiragana", "index": index, "raw_response": text}
        surface_cursor += len(surface)
        reading_cursor += len(reading)
    if "input_reading" not in data or "canonical_reference" not in data:
        return {"ok": False, "error": "reading_fields_invalid", "raw_response": text}
    if "".join(t["reading"] for t in data["tokens"]) != data["input_reading"]:
        return {"ok": False, "error": "reading_concat_mismatch", "raw_response": text}
    if "".join(t["surface"] for t in data["tokens"]) != data["canonical_reference"]:
        return {"ok": False, "error": "surface_concat_mismatch", "raw_response": text}
    return {"ok": True, "data": data, "raw_response": text}


def _dry_response(case: Mapping[str, Any], mecab_tokens: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source = str(case.get("source_sentence", case.get("source_text", case.get("canonical_reference", ""))))
    tokens = [dict(token) for token in mecab_tokens]
    return {"schema_version": SCHEMA_VERSION, "case_id": str(case.get("case_id", case.get("id", ""))),
            "input_reading": "".join(str(t["reading"]) for t in tokens),
            "canonical_reference": source, "tokens": tokens,
            "needs_review": False, "review_reasons": []}


def _gold_case(case: Mapping[str, Any], annotation: Mapping[str, Any], raw_hash: str) -> dict[str, Any]:
    result = dict(case)
    result["input"] = annotation["input_reading"]
    result["canonical_reference"] = annotation["canonical_reference"]
    result["references"] = list(case.get("references", [annotation["canonical_reference"]]))
    result["gold_paths"] = [{"path_id": "canonical", "is_canonical": True, "tokens": annotation["tokens"]}]
    result["annotation_status"] = "annotated"
    result["annotation_provenance"] = {"provider": "anthropic", "model_snapshot": "",
        "prompt_sha256": "", "schema_sha256": "", "request_parameters": {"temperature": 0, "seed": SEED, "top_p": 1},
        "raw_response_sha256": raw_hash, "response_id": "", "checker": {"provider": "", "model_snapshot": "", "raw_response_sha256": "", "status": "not_run"},
        "human_audit": [], "adjudication": {"required": bool(annotation.get("needs_review", False)), "status": "open", "history": []}}
    result["span_tokens"] = sum(not t["is_literal"] for t in annotation["tokens"])
    result["reading_length"] = len(annotation["input_reading"])
    result["input_length"] = len(annotation["input_reading"])
    return result


def run_annotation(candidates: Sequence[Mapping[str, Any]], output_dir: str | Path, model: str = "claude-opus-5-20250514", *, dry_run: bool = False, backend: str = "api", ssh_host: str = "ws") -> list[dict[str, Any]]:
    """Annotate candidates and write one JSONL raw artifact plus failure log."""
    import hashlib
    if backend == "claude-cli" and model == "claude-opus-5-20250514":
        model = "opus"
    root = Path(output_dir)
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_dir / f"responses-{stamp}.jsonl"
    failures: list[dict[str, Any]] = []
    output: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_file:
        if backend not in {"api", "dry-run", "claude-cli"}:
            raise ValueError(f"unknown annotation backend: {backend}")
        if dry_run:
            backend = "dry-run"
        candidate_list = list(candidates)
        batches = ([candidate_list[i:i + 50] for i in range(0, len(candidate_list), 50)]
                   if backend == "claude-cli" else [[case] for case in candidates])
        for batch in batches:
            if backend == "claude-cli":
                case_ids = [str(case.get("case_id", case.get("id", ""))) for case in batch]
                try:
                    annotations = call_claude_cli_batch([dict(case) for case in batch], model=model, ssh_host=ssh_host)
                except Exception as exc:
                    failures.extend({"case_id": case_id, "error": "request_failed", "detail": str(exc)} for case_id in case_ids)
                    continue
                raw_text = json.dumps(annotations, ensure_ascii=False, separators=(",", ":"))
                digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
                raw_file.write(json.dumps({"case_ids": case_ids, "timestamp": stamp, "model": model,
                                           "raw_response": raw_text, "sha256": digest}, ensure_ascii=False) + "\n")
                if len(annotations) != len(batch):
                    failures.extend({"case_id": case_id, "error": "batch_length_mismatch"} for case_id in case_ids)
                    continue
                response_items = zip(batch, annotations)
            else:
                response_items = ((batch[0], None),)

            for case, batch_annotation in response_items:
                case_id = str(case.get("case_id", case.get("id", "")))
                mecab = list(case.get("mecab_tokens", case.get("gold_tokens", [])))
                if backend == "dry-run":
                    raw_text = json.dumps(_dry_response(case, mecab), ensure_ascii=False, separators=(",", ":"))
                elif backend == "claude-cli":
                    raw_text = json.dumps(batch_annotation, ensure_ascii=False, separators=(",", ":"))
                else:
                    built = build_llm_messages(case, mecab, case.get("d1_anchors", []))
                    try:
                        response = call_llm(built, model=model)
                        raw_text = _text_from_response(response)
                    except Exception as exc:
                        failures.append({"case_id": case_id, "error": "request_failed", "detail": str(exc)})
                        continue
                digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
                if backend != "claude-cli":
                    raw_file.write(json.dumps({"case_id": case_id, "timestamp": stamp, "model": model, "raw_response": raw_text, "sha256": digest}, ensure_ascii=False) + "\n")
                parsed = parse_llm_response(raw_text)
                if not parsed["ok"]:
                    failures.append({"case_id": case_id, "error": parsed.get("error"), "detail": parsed})
                    continue
                annotation = parsed["data"]
                expected_source = str(case.get("source_sentence", case.get("source_text", case.get("canonical_reference", ""))))
                if annotation.get("case_id") != case_id or annotation.get("canonical_reference") != expected_source:
                    failures.append({"case_id": case_id, "error": "source_identity_mismatch",
                                      "detail": {"response_case_id": annotation.get("case_id"),
                                                 "response_reference": annotation.get("canonical_reference")}})
                    continue
                output.append(_gold_case(case, annotation, digest))
    (root / "annotation_failures.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in failures) + ("\n" if failures else ""), encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="use MeCab provisional tokens; never call a backend")
    parser.add_argument("--backend", choices=("api", "claude-cli", "dry-run"), default="api")
    parser.add_argument("--input", type=Path, help="candidate JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("data/e0/annotations"))
    parser.add_argument("--model", default="claude-opus-5-20250514")
    args = parser.parse_args(argv)
    if args.input:
        rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
        run_annotation(rows, args.output_dir, args.model, dry_run=args.dry_run, backend=args.backend)
    elif not args.dry_run:
        parser.error("--dry-run or --input is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
