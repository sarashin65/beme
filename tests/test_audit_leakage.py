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


def test_failed_check_exits_nonzero_and_is_named(tmp_path: Path) -> None:
    """A check that ran and came back bad must fail the audit, not just log."""
    module = load_module()
    d1_dir = tmp_path / "d1"
    d1_dir.mkdir()
    manifest = json.loads((ROOT / "data" / "d1" / "manifest.json").read_text(encoding="utf-8"))
    # More unique articles than records is impossible; the audit must say so.
    manifest["unique_articles"]["test"] = manifest["counts"]["test"] + 1
    (d1_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (d1_dir / "data_gate.json").write_text(
        (ROOT / "data" / "d1" / "data_gate.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    output = tmp_path / "audit.json"
    code = module.main([
        "--d1-dir", str(d1_dir),
        "--ajimee-cases", str(ROOT / "data" / "ajimee_bench_jwtd_v2_v1.json"),
        "--mozc-cases", str(ROOT / "data" / "mozc_oss.tsv"),
        "--cases", str(ROOT / "data" / "evaluation_cases.jsonl"),
        "--output", str(output),
    ])
    document = json.loads(output.read_text(encoding="utf-8"))
    assert code == 1
    assert document["failures"] == ["manifest_consistency:unique_articles_le_counts"]


def test_unverifiable_checks_do_not_fail_the_audit(tmp_path: Path) -> None:
    module = load_module()
    output = tmp_path / "audit.json"
    code = module.main([
        "--d1-dir", str(ROOT / "data" / "d1"),
        "--ajimee-cases", str(ROOT / "data" / "ajimee_bench_jwtd_v2_v1.json"),
        "--mozc-cases", str(ROOT / "data" / "mozc_oss.tsv"),
        "--cases", str(ROOT / "data" / "evaluation_cases.jsonl"),
        "--output", str(output),
    ])
    document = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert document["unverifiable"] and document["failures"] == []
    # The unavailable branch must not hand every suite the same inner dicts.
    overlap = document["evaluation_overlap"]
    assert overlap["ajimee"]["document_leakage"]["counts_as_leakage"] is True
    assert overlap["mozc"]["reading_overlap"]["counts_as_leakage"] is False
