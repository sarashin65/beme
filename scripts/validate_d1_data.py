#!/usr/bin/env python3
"""Validate the mechanical D1 data gates and emit a machine-readable report."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data" / "d1"
TARGETS = {"train": 10_000, "dev": 1_000, "test": 2_000}


def normalized_fingerprint(group: dict[str, Any]) -> str:
    text = unicodedata.normalize(
        "NFKC",
        f"{group['left_context']}\0{group['reading']}\0{group['positive']}\0{group['right_context']}",
    )
    normalized = "".join(character for character in text.casefold() if not character.isspace())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def validate(data_dir: Path) -> dict[str, Any]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    splits = {name: load_jsonl(data_dir / f"{name}.jsonl") for name in TARGETS}
    article_ids = {name: {str(group["article_id"]) for group in rows} for name, rows in splits.items()}
    fingerprints = {name: {normalized_fingerprint(group) for group in rows} for name, rows in splits.items()}
    invalid_groups: list[str] = []
    for split, rows in splits.items():
        for group in rows:
            candidates = group.get("candidates")
            costs = group.get("viterbi_costs")
            if (
                group.get("split") != split
                or not isinstance(candidates, list)
                or group.get("positive") not in candidates
                or not isinstance(costs, list)
                or len(candidates) != len(costs)
                or len(candidates) < 2
                or int(group.get("gold_index", -1)) != candidates.index(group["positive"])
            ):
                invalid_groups.append(str(group.get("id")))
    article_overlaps = {
        "train_dev": sorted(article_ids["train"] & article_ids["dev"]),
        "train_test": sorted(article_ids["train"] & article_ids["test"]),
        "dev_test": sorted(article_ids["dev"] & article_ids["test"]),
    }
    fingerprint_overlaps = {
        "train_dev": len(fingerprints["train"] & fingerprints["dev"]),
        "train_test": len(fingerprints["train"] & fingerprints["test"]),
        "dev_test": len(fingerprints["dev"] & fingerprints["test"]),
    }
    audit = manifest.get("audit", {})
    gates = {
        "counts_exact": all(len(splits[name]) == target for name, target in TARGETS.items()),
        "article_disjoint": not any(article_overlaps.values()),
        "normalized_group_disjoint": not any(fingerprint_overlaps.values()),
        "all_groups_valid": not invalid_groups,
        "evaluation_overlap_exclusion_recorded": bool(manifest.get("evaluation_overlap")),
        "source_manifest_complete": all(
            manifest.get("source", {}).get(field)
            for field in ("dataset", "config", "revision", "license")
        ),
        "audit_complete": audit.get("completed") is True,
        "audit_accuracy_at_least_98pct": float(audit.get("label_accuracy", 0.0)) >= 0.98,
        "train_gold_in_candidate_at_least_10k": len(splits["train"]) >= 10_000 and not invalid_groups,
        "context_contrast_present": int(manifest.get("contrast_pairs", {}).get("test", 0)) > 0,
    }
    report = {
        "counts": {name: len(rows) for name, rows in splits.items()},
        "unique_articles": {name: len(values) for name, values in article_ids.items()},
        "unique_readings": {name: len({group["reading"] for group in rows}) for name, rows in splits.items()},
        "article_overlaps": article_overlaps,
        "normalized_group_fingerprint_overlaps": fingerprint_overlaps,
        "invalid_group_count": len(invalid_groups),
        "invalid_group_ids": invalid_groups[:20],
        "contrast_pairs": manifest.get("contrast_pairs"),
        "audit": audit,
        "gates": gates,
        "data_go": all(gates.values()),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.data_dir)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["data_go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
