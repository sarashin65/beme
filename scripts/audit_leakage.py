#!/usr/bin/env python3
"""Audit d1 manifest consistency and evaluation-data overlap independently."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beme.benchmark import load_ajimee_cases, load_mozc_cases  # noqa: E402
from beme.evaluation import load_cases  # noqa: E402
from beme.training_data import TrainingExample, load_jsonl  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def split_for_article(article_id: str, seed: int, ranges: list[list[Any]]) -> str | None:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{article_id}".encode()).digest(), "big") % 10000
    lower = 0
    for name, upper in ranges:
        if lower <= value < int(upper):
            return str(name)
        lower = int(upper)
    return None


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def audit_manifest(d1_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = d1_dir / "manifest.json"
    gate_path = d1_dir / "data_gate.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    counts = manifest["counts"]
    unique_articles = manifest["unique_articles"]
    unique_readings = manifest["unique_readings"]
    checks.append(_check("unique_articles_le_counts", all(unique_articles[k] <= counts[k] for k in counts), "unique article counts do not exceed split counts"))
    checks.append(_check("unique_readings_le_counts", all(unique_readings[k] <= counts[k] for k in counts), "unique reading counts do not exceed split counts"))
    checks.append(_check("contrast_pairs_le_counts", all(manifest["contrast_pairs"][k] <= counts[k] for k in counts), "contrast-pair counts do not exceed split counts"))
    checks.append(_check("gate_counts_match_manifest", gate["counts"] == counts, "data_gate counts equal manifest counts"))
    checks.append(_check("audit_accuracy_recomputed", manifest["audit"]["label_accuracy"] == manifest["audit"]["correct"] / manifest["audit"]["audited"], "label_accuracy equals correct/audited"))
    checks.append(_check("audit_incorrect_recomputed", manifest["audit"]["incorrect"] == manifest["audit"]["audited"] - manifest["audit"]["correct"], "incorrect equals audited-correct"))
    overlaps = gate["article_overlaps"]
    checks.append(_check("article_disjoint_gate", gate["gates"]["article_disjoint"] == all(not overlaps[name] for name in overlaps), "article_disjoint agrees with article_overlaps"))
    normalized = gate["normalized_group_fingerprint_overlaps"]
    checks.append(_check("normalized_group_gate", gate["gates"]["normalized_group_disjoint"] == all(value == 0 for value in normalized.values()), "normalized_group_disjoint agrees with overlap counts"))
    checks.append(_check("audit_complete_gate", gate["gates"]["audit_complete"] == bool(manifest["audit"]["completed"]), "audit_complete agrees with audit.completed"))
    checks.append(_check("audit_accuracy_gate", gate["gates"]["audit_accuracy_at_least_98pct"] == (manifest["audit"]["label_accuracy"] >= 0.98), "audit accuracy gate agrees with label_accuracy"))
    checks.append(_check("invalid_groups_gate", gate["gates"]["all_groups_valid"] == (gate["invalid_group_count"] == 0), "all_groups_valid agrees with invalid_group_count"))
    checks.append(_check("contrast_present_gate", gate["gates"]["context_contrast_present"] == (sum(manifest["contrast_pairs"].values()) > 0), "context_contrast_present agrees with contrast counts"))
    checks.append(_check("source_metadata_present", all(key in manifest["source"] for key in ("dataset", "revision", "config")), "required source metadata is present"))
    return {"checks": checks, "all_passed": all(item["passed"] for item in checks)}, manifest


def _d1_examples(d1_dir: Path) -> tuple[dict[str, tuple[TrainingExample, ...]], bool]:
    paths = {name: d1_dir / f"{name}.jsonl" for name in ("train", "dev", "test")}
    available = all(path.is_file() for path in paths.values())
    return ({name: load_jsonl(path) for name, path in paths.items()} if available else {}), available


def audit_split_recomputation(d1_dir: Path) -> dict[str, Any]:
    manifest = json.loads((d1_dir / "manifest.json").read_text(encoding="utf-8"))
    examples, available = _d1_examples(d1_dir)
    if not available:
        return {"verified": False, "reason": "train.jsonl/dev.jsonl/test.jsonl are not available"}
    mismatches = []
    for split, rows in examples.items():
        for example in rows:
            if example.article_id is None or split_for_article(example.article_id, manifest["split"]["seed"], manifest["split"]["ranges"]) != split:
                mismatches.append(example.id or "<missing-id>")
    return {"verified": not mismatches, "reason": "all article IDs recomputed from seed and ranges" if not mismatches else f"mismatched records: {mismatches[:5]}", "mismatches": mismatches}


def _evaluation_material(ajimee_path: Path, mozc_path: Path, cases_path: Path) -> dict[str, dict[str, Any]]:
    ajimee = load_ajimee_cases(ajimee_path)
    mozc = load_mozc_cases(mozc_path)
    handmade = load_cases(cases_path)
    return {
        "ajimee": {"readings": {case.input for case in ajimee}, "surfaces": {surface for case in ajimee for surface in case.references}, "documents": {_normalized(case.context + "".join(case.references)) for case in ajimee}},
        "mozc": {"readings": {case.reading for case in mozc}, "surfaces": {case.expected for case in mozc}, "documents": set()},
        "handmade": {"readings": {case["reading"] for case in handmade}, "surfaces": {surface for case in handmade for surface in case["judgments"]}, "documents": {_normalized(case["preceding_context"] + case["following_context"] + "".join(case["judgments"])) for case in handmade}},
    }


def audit_evaluation_overlap(d1_dir: Path, ajimee_path: Path, mozc_path: Path, cases_path: Path) -> tuple[dict[str, Any], bool]:
    evaluation = _evaluation_material(ajimee_path, mozc_path, cases_path)
    d1, available = _d1_examples(d1_dir)
    if not available:
        empty = {"document_leakage": {"verified": False, "matched": None, "reason": "d1 JSONL files are unavailable"}, "reading_overlap": {"verified": False, "count": None, "rate": None, "is_leakage": False}, "vocabulary_overlap": {"verified": False, "matched": None, "total": None, "rate": None, "is_leakage": False}}
        return {name: dict(empty) for name in evaluation}, False
    d1_readings = {example.reading for rows in d1.values() for example in rows}
    train_surfaces = {example.target_surface for example in d1["train"]}
    d1_ngrams = set()
    for example in d1["train"]:
        for text in (example.target_surface, example.left_context, example.right_context):
            normalized = _normalized(text)
            d1_ngrams.update(normalized[index:index + 12] for index in range(max(0, len(normalized) - 11)))
    result = {}
    for name, values in evaluation.items():
        eval_ngrams = {gram for text in values["documents"] for gram in (text[index:index + 12] for index in range(max(0, len(text) - 11)))}
        reading_count = len(values["readings"] & d1_readings)
        vocabulary_count = len(values["surfaces"] & train_surfaces)
        result[name] = {
            "document_leakage": {"verified": True, "matched": len(eval_ngrams & d1_ngrams), "evaluation_ngrams": len(eval_ngrams), "is_leakage": True},
            "reading_overlap": {"verified": True, "count": reading_count, "total": len(values["readings"]), "rate": reading_count / len(values["readings"]) if values["readings"] else 0.0, "is_leakage": False},
            "vocabulary_overlap": {"verified": True, "matched": vocabulary_count, "total": len(values["surfaces"]), "rate": vocabulary_count / len(values["surfaces"]) if values["surfaces"] else 0.0, "is_leakage": False},
        }
    return result, True


def audit_mozc_claim(path: Path) -> dict[str, Any]:
    cases = load_mozc_cases(path)
    text = path.read_text(encoding="utf-8")
    first_line = text.splitlines()[0] if text else ""
    extra = sorted({field for field in ("original_text", "context", "context_text", "document") if field in first_line.split("\t")})
    fields = ["id", "label", "reading", "expected", "command", "expected_rank", *extra]
    return {"claim": "Mozc contains readings and assertions but no original text or context", "verified": not extra, "fields_present": fields, "case_count": len(cases), "extra_source_fields": extra}


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".leak-audit-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-dir", type=Path, required=True)
    parser.add_argument("--ajimee-cases", type=Path, required=True)
    parser.add_argument("--mozc-cases", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    consistency, _ = audit_manifest(args.d1_dir)
    split = audit_split_recomputation(args.d1_dir)
    overlap, available = audit_evaluation_overlap(args.d1_dir, args.ajimee_cases, args.mozc_cases, args.cases)
    mozc_claim = audit_mozc_claim(args.mozc_cases)
    unverifiable = []
    if not available:
        unverifiable.extend(["split field recomputation against article IDs", "d1-vs-evaluation document n-gram overlap", "d1-vs-evaluation reading overlap", "d1 train vocabulary overlap"])
    document = {"manifest_consistency": consistency, "split_recomputation": split, "evaluation_overlap": overlap, "mozc_source_document_claim": mozc_claim, "d1_jsonl_available": available, "unverifiable": unverifiable}
    _atomic_json(args.output, document)
    if args.json:
        print(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
