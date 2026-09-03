#!/usr/bin/env python3
"""Compute the task6 metrics (overall / non-contrast / contrast / both_correct)
from rerank_eval score outputs, per condition and seed.

Reads artifacts/issue19-f2-contrast/<cond>/seed<seed>/scores.jsonl
(first line = provenance, rest = records).
"""
from __future__ import annotations

from pathlib import Path
import collections
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR = ROOT / "artifacts" / "issue19-f2-contrast"
SEEDS = (9, 17, 33)


def load_records(condition: str, seed: int) -> list[dict]:
    path = ROOT_DIR / condition / f"seed{seed}" / "scores.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines[1:]]  # skip provenance
    assert len(records) == 2000, f"{condition} seed{seed}: {len(records)} records"
    return records


def metrics(records: list[dict]) -> dict:
    correct = {}
    pairs: dict[str, list[bool]] = collections.defaultdict(list)
    for r in records:
        top1 = r["candidates"][0]["surface"]
        ok = top1 == r["gold"]
        correct[r["id"]] = ok
        if r.get("contrast_pair_id"):
            pairs[r["contrast_pair_id"]].append(ok)
    overall = sum(correct.values()) / len(correct)
    contrast_ids = {r["id"] for r in records if r.get("contrast_pair_id")}
    non_contrast = [correct[i] for i in correct if i not in contrast_ids]
    contrast = [correct[i] for i in correct if i in contrast_ids]
    both = sum(1 for p in pairs.values() if all(p))
    return {
        "overall_top1": overall,
        "overall_errors": len(correct) - sum(correct.values()),
        "non_contrast_top1": sum(non_contrast) / len(non_contrast) if non_contrast else float("nan"),
        "non_contrast_n": len(non_contrast),
        "contrast_top1": sum(contrast) / len(contrast) if contrast else float("nan"),
        "contrast_n": len(contrast),
        "both_correct": both,
        "pairs_total": len(pairs),
    }


def main() -> int:
    print(f"{'condition':4s} {'seed':3s} {'overall':>8s} {'err':>4s} {'non_cont':>8s} {'cont':>8s} {'both':>5s}")
    for cond in ("A8", "C0"):
        for seed in SEEDS:
            m = metrics(load_records(cond, seed))
            print(f"{cond:4s} {seed:3d} {m['overall_top1']:.6f} {m['overall_errors']:4d} "
                  f"{m['non_contrast_top1']:.6f} {m['contrast_top1']:.6f} "
                  f"{m['both_correct']:3d}/{m['pairs_total']}")
    print("\n=== 中央値サマリ ===")
    for cond in ("A8", "C0"):
        rows = [metrics(load_records(cond, seed)) for seed in SEEDS]
        med = lambda key: statistics.median(m[key] for m in rows)
        errs = [m["overall_errors"] for m in rows]
        print(f"{cond}: overall_median={med('overall_top1'):.6f} errors={errs} "
              f"non_contrast_median={med('non_contrast_top1'):.6f} "
              f"contrast_median={med('contrast_top1'):.6f} "
              f"both_correct={[m['both_correct'] for m in rows]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
