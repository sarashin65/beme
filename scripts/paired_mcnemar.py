#!/usr/bin/env python3
"""Compute the C0-vs-A8 (and C1-vs-A0) paired McNemar from diagnose_d1.json cases.

The run_ablations summary only compares every condition against A0; the
mission's preregistered decision for C0 is against A8, so we rebuild the
paired table from per-case correctness (id-keyed, same test set).
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beme.rerank_stats import mcnemar  # noqa: E402

ROOT_DIR = ROOT / "artifacts" / "issue19-f2"
SEEDS = (9, 17, 33)


def per_case_correct(condition: str, seed: int) -> dict[str, bool]:
    path = ROOT_DIR / condition / f"seed{seed}" / "diagnose_d1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {case["id"]: case["learned"]["correct"] for case in data["cases"]}


def paired(cond_a: str, cond_b: str, seed: int) -> tuple[int, int, int, int]:
    """b = A correct & B wrong, c = B correct & A wrong, plus counts."""
    a = per_case_correct(cond_a, seed)
    b_map = per_case_correct(cond_b, seed)
    assert set(a) == set(b_map), f"case id mismatch: {cond_a} vs {cond_b} seed{seed}"
    b = sum(1 for i in a if a[i] and not b_map[i])
    c = sum(1 for i in a if b_map[i] and not a[i])
    return b, c


def main() -> int:
    print("=== C0 vs A8 (本命・事前登録判定) ===")
    for seed in SEEDS:
        b, c = paired("A8", "C0", seed)
        res = mcnemar(b, c)
        print(f"  seed{seed}: b(A8正解&C0誤り)={b} c(C0正解&A8誤り)={c} "
              f"p={res['p_value']:.3e} method={res['method']}")

    print("=== C1 vs A0 ===")
    for seed in SEEDS:
        b, c = paired("A0", "C1", seed)
        res = mcnemar(b, c)
        print(f"  seed{seed}: b(A0正解&C1誤り)={b} c(C1正解&A0誤り)={c} "
              f"p={res['p_value']:.3e} method={res['method']}")

    print("=== 誤り件数（diagnose から再確認）===")
    for cond in ("A0", "A8", "C0", "C1"):
        errors = []
        for seed in SEEDS:
            correct = per_case_correct(cond, seed)
            errors.append(len(correct) - sum(correct.values()))
        print(f"  {cond}: {errors} median={sorted(errors)[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
