"""Tests for independent leakage-audit helpers and CLI."""

from __future__ import annotations

from pathlib import Path
import json
import importlib.util
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def load_module():
    path = ROOT / "scripts" / "audit_leakage.py"
    spec = importlib.util.spec_from_file_location("audit_leakage", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repo_partial_audit_marks_d1_jsonl_unverifiable(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    subprocess.run([sys.executable, "scripts/audit_leakage.py", "--d1-dir", "data/d1", "--ajimee-cases", "data/ajimee_bench_jwtd_v2_v1.json", "--mozc-cases", "data/mozc_oss.tsv", "--cases", "data/evaluation_cases.jsonl", "--output", str(output), "--json"], cwd=ROOT, check=True)
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["manifest_consistency"]["all_passed"]
    assert document["d1_jsonl_available"] is False
    assert document["unverifiable"]


def test_mozc_claim_rejects_source_document_column(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "mozc.tsv"
    path.write_text("id\tlabel\treading\texpected\tcommand\toriginal_text\n", encoding="utf-8")
    result = module.audit_mozc_claim(path)
    assert result["verified"] is False
    assert "original_text" in result["extra_source_fields"]


def test_split_recomputation_is_deterministic() -> None:
    module = load_module()
    assert module.split_for_article("article-1", 20260822, [["train", 7700], ["dev", 8500], ["test", 10000]]) == module.split_for_article("article-1", 20260822, [["train", 7700], ["dev", 8500], ["test", 10000]])
