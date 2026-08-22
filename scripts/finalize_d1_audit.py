#!/usr/bin/env python3
"""Validate a completed D1 audit sample and update its data manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data" / "d1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(data_dir: Path) -> dict[str, object]:
    audit_path = data_dir / "audit.jsonl"
    manifest_path = data_dir / "manifest.json"
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) < 200:
        raise ValueError(f"audit must contain at least 200 groups, got {len(rows)}")
    invalid = [index for index, row in enumerate(rows, 1) if row.get("audit_label_correct") not in {True, False}]
    if invalid:
        raise ValueError(f"audit labels must be boolean; incomplete rows: {invalid[:10]}")
    correct = sum(bool(row["audit_label_correct"]) for row in rows)
    accuracy = correct / len(rows)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts_sha256"]["audit"] = file_sha256(audit_path)
    manifest["audit"].update(
        {
            "completed": True,
            "audited": len(rows),
            "correct": correct,
            "incorrect": len(rows) - correct,
            "label_accuracy": accuracy,
            "data_gate_pass": accuracy >= 0.98,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest["audit"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    print(json.dumps(finalize(args.data_dir), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
