#!/usr/bin/env python3
"""Run and aggregate the Phase 3 ablation matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beme.features import registered_spec_ids  # noqa: E402
from beme.rerank_stats import mcnemar  # noqa: E402


CONDITIONS = {
    "A0": "decomposed_v1",
    "A1": "decomposed_v1-lex_only",
    "A2": "decomposed_v1-trans_only",
    "A3": "decomposed_v1-no_ipadic_cost",
    "A4": "decomposed_v1-no_source",
    "A5": "decomposed_v1-no_katakana",
    "A6": "decomposed_v1-no_shape",
    "M0": "decomposed_v1+matrix_v1",
    "M1": "decomposed_v1+matrix_v1-pos_pairs",
    "F0": "decomposed_v1+freq_v1",
    "F1": "decomposed_v1+freq_v1-no_ipadic_cost",
    "X0": "decomposed_v1+matrix_v1+freq_v1",
}
SEEDS = (9, 17, 33)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], cwd: Path, stdout: Path | None = None) -> None:
    if stdout is None:
        subprocess.run(command, cwd=cwd, check=True)
        return
    with stdout.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)


def summarize_values(values: list[float | int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"median": statistics.median(ordered), "min": min(ordered), "max": max(ordered), "range": [min(ordered), max(ordered)]}


def _normalized_weights(path: Path) -> dict[str, float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    values = {name: float(value) for name, value in document.get("weights", {}).items()}
    scale = values.get("lex.ipadic_cost_mapped", 0.0)
    return {name: value / scale for name, value in sorted(values.items())} if scale else {}


def _gate(report: dict[str, Any], evaluate: dict[str, Any]) -> dict[str, Any]:
    selected = report["selected_model"]["test"]
    baseline = report["baseline"]["test"]
    handmade = evaluate["handmade"]["top1_accuracy"]
    ajimee = evaluate["ajimee"]["mean_cer"]
    mozc = (evaluate["mozc"]["passed_count"], evaluate["mozc"]["evaluated_count"])
    checks = {
        "test_model_errors": (selected["model_errors"], 538, 1),
        "test_exact_matches": (selected["exact_matches"], 1462, 1),
        "test_coverage_errors": (selected["coverage_errors"], 0, 0),
        "baseline_test_model_errors": (baseline["model_errors"], 620, 0),
        "ajimee_cer": (ajimee, 0.287092, 0),
        "mozc_passed": (mozc, (87, 498), 0),
        "handmade_top1": (handmade, 13 / 14, 0),
    }
    passed = {
        "test_model_errors": abs(checks["test_model_errors"][0] - 538) <= 1,
        "test_exact_matches": abs(checks["test_exact_matches"][0] - 1462) <= 1,
        "test_coverage_errors": checks["test_coverage_errors"][0] == 0,
        "baseline_test_model_errors": checks["baseline_test_model_errors"][0] == 620,
        "ajimee_cer": math.isclose(checks["ajimee_cer"][0], 0.287092, abs_tol=1e-4),
        "mozc_passed": mozc == (87, 498),
        "handmade_top1": math.isclose(handmade, 13 / 14, abs_tol=1e-12),
    }
    return {"passed": all(passed.values()), "checks": {name: {"actual": value[0], "expected": value[1], "passed": passed[name]} for name, value in checks.items()}}


def _condition_summary(root: Path, condition: str, seeds: list[int]) -> dict[str, Any]:
    runs = []
    for seed in seeds:
        directory = root / condition / f"seed{seed}"
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        evaluate_path = directory / "evaluate.json"
        diagnose_path = directory / "diagnose_d1.json"
        evaluate = json.loads(evaluate_path.read_text(encoding="utf-8")) if evaluate_path.read_text(encoding="utf-8").lstrip().startswith("{") else None
        diag = json.loads(diagnose_path.read_text(encoding="utf-8")) if diagnose_path.is_file() else None
        runs.append({"seed": seed, "test": report["selected_model"]["test"], "evaluate": evaluate, "diagnose": diag, "normalized_weights": _normalized_weights(directory / "weights.json"), "complete": evaluate is not None and diag is not None})
    model_errors = [run["test"]["end_to_end_errors"] for run in runs]
    # The contingency table is (A0 correct/wrong) x (condition correct/wrong)
    # over the *learned* model of each, paired by seed: the ablation question
    # is whether the condition differs from A0, not whether training moved the
    # untrained baseline (which is the same document for every condition and
    # would make even a condition identical to A0 look significant).
    a0_runs = {}
    for seed in seeds:
        path = root / "A0" / f"seed{seed}" / "diagnose_d1.json"
        if path.is_file():
            a0_runs[seed] = json.loads(path.read_text(encoding="utf-8"))
    tables = []
    for run in runs:
        a0 = a0_runs.get(run["seed"])
        if not run["diagnose"] or not a0:
            continue
        base = {item["id"]: item["learned"]["correct"] for item in a0["cases"]}
        current = {item["id"]: item["learned"]["correct"] for item in run["diagnose"]["cases"]}
        shared = sorted(set(base) & set(current))
        b = sum(base[key] and not current[key] for key in shared)
        c = sum(not base[key] and current[key] for key in shared)
        tables.append({
            "seed": run["seed"],
            "compared": len(shared),
            "unpaired": len(set(base) ^ set(current)),
            "table": mcnemar(b, c),
        })
    return {"seeds": runs, "test_end_to_end_errors": summarize_values(model_errors), "mcnemar": tables}


def summarize_only(root: Path) -> dict[str, Any]:
    available = [condition for condition in CONDITIONS if all((root / condition / f"seed{seed}" / "report.json").is_file() for seed in SEEDS)]
    conditions = {condition: _condition_summary(root, condition, list(SEEDS)) for condition in available}
    incomplete = [
        {"condition": condition, "seed": run["seed"], "reason": "downstream artifact missing after subprocess failure"}
        for condition, value in conditions.items() for run in value["seeds"] if not run["complete"]
    ]
    a0_errors = conditions.get("A0", {}).get("test_end_to_end_errors", {}).get("median")
    for condition, value in conditions.items():
        median = value["test_end_to_end_errors"]["median"]
        # Every recorded seed must clear the threshold, not one representative
        # seed: what changes between seeds is the model, not the sample, so a
        # difference only one seed of three can see is not one the condition
        # reproduces.  The weakest seed is what the label rests on, so it is
        # recorded next to it.
        p_values = [item["table"]["p_value"] for item in value["mcnemar"]]
        weakest = max(p_values) if p_values else None
        if condition == "A0":
            # A0 against itself is b = c = 0 by construction; the label is
            # stated rather than derived so the self-comparison is explicit.
            label = "indeterminate"
        elif weakest is None or a0_errors is None or weakest >= 0.05:
            label = "indeterminate"
        elif median == a0_errors:
            # A significant table with no median difference does not order the
            # two conditions, so neither direction may be claimed.
            label = "indeterminate"
        elif median > a0_errors:
            label = "significant_worse"
        else:
            label = "significant_better"
        value["decision_label"] = label
        value["decision"] = {
            "label": label,
            "rule": "every recorded seed p < 0.05 and the median differs",
            "seeds_compared": len(p_values),
            "max_p_value": weakest,
            "min_p_value": min(p_values) if p_values else None,
        }
    return {
        "conditions": conditions,
        "completed_conditions": available,
        "failed": incomplete,
        "skipped": [condition for condition in CONDITIONS if condition not in available],
        "interpretation_notes": {
            "decision_basis": "d1 test McNemar result and end-to-end error median; suite metrics are descriptive only",
            "mcnemar_comparison": "learned condition against learned A0 on the same seed; b = A0 correct and condition wrong, c = the reverse",
            "seed_agreement": "a label requires every recorded seed to clear p < 0.05; one seed out of three is not a reproducible difference, and the tables must not be pooled because every seed scores the same 2,000 test items",
            "multiple_comparisons": "one test per condition per seed, with no correction applied; a p just under 0.05 on a single condition is within chance for this many tests",
            "linear_dependency": "lex.script.* and trans.* both count one token per path, so raw weights are not feature importance",
            "reading_length": "lex.reading_len is structurally constant for a reading and is not learned",
            "transition_only": "A2 tie-heavy one-token behavior is an expected ablation lower bound, not an error",
        },
    }


def _run_one(root: Path, condition: str, seed: int, force: bool) -> dict[str, Any]:
    directory = root / condition / f"seed{seed}"
    if directory.exists() and not force:
        raise FileExistsError(f"output already exists: {directory}")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    spec = CONDITIONS[condition]
    weights = ROOT / "data" / "weights" / (
        Path("decomposed_v1.json")
        if condition == "A0"
        else Path("ablations") / f"{spec}.json"
    )
    learned = directory / "weights.json"
    report = directory / "report.json"
    started = time.monotonic()
    _run([sys.executable, "scripts/train_scorer.py", "--train", "data/d1/train.jsonl", "--dev", "data/d1/dev.jsonl", "--test", "data/d1/test.jsonl", "--dictionary", "data/dictionary_v2.tsv", "--initial-weights", str(weights), "--epochs", "5", "--learning-rate", "0.1", "--seed", str(seed), "--output-weights", str(learned), "--output-report", str(report)], ROOT)
    _run([sys.executable, "scripts/evaluate.py", "--suite", "all", "--reranker", "mock", "--dictionary", "data/dictionary_v2.tsv", "--weights", str(learned), "--json"], ROOT, directory / "evaluate.json")
    for weight_label, weight_path in (("baseline", ROOT / "data" / "weights" / "decomposed_v1.json"), ("learned", learned)):
      for name, data, output in (("d1", "data/d1/test.jsonl", "d1"), ("mozc", None, "mozc"), ("handmade", None, "handmade")):
        command = [sys.executable, "scripts/oracle_recall.py", "--dictionary", "data/dictionary_v2.tsv", "--weights", str(learned), "--output", str(directory / "oracle" / f"{output}.json")]
        if data:
            command += ["--data", data]
        else:
            command += ["--suite", name]
        command[command.index(str(learned))] = str(weight_path)
        command[command.index(str(directory / "oracle" / f"{output}.json"))] = str(directory / "oracle" / f"{weight_label}-{output}.json")
        (directory / "oracle").mkdir(exist_ok=True)
        _run(command, ROOT)
    _run([sys.executable, "scripts/diagnose_cases.py", "--data", "data/d1/test.jsonl", "--stratify", "--dictionary", "data/dictionary_v2.tsv", "--baseline-weights", "data/weights/decomposed_v1.json", "--learned-weights", str(learned), "--output", str(directory / "diagnose_d1.json")], ROOT)
    _run([sys.executable, "scripts/diagnose_cases.py", "--suite", "handmade", "--dictionary", "data/dictionary_v2.tsv", "--baseline-weights", "data/weights/decomposed_v1.json", "--learned-weights", str(learned), "--output", str(directory / "diagnose_handmade.json")], ROOT)
    _run([sys.executable, "scripts/diagnose_cases.py", "--suite", "mozc", "--dictionary", "data/dictionary_v2.tsv", "--baseline-weights", "data/weights/decomposed_v1.json", "--learned-weights", str(learned), "--output", str(directory / "diagnose_mozc.json")], ROOT)
    _run([sys.executable, "scripts/measure_cost_scale.py", "--data", "data/d1/test.jsonl", "--dictionary", "data/dictionary_v2.tsv", "--baseline-weights", "data/weights/decomposed_v1.json", "--learned-weights", str(learned), "--output", str(directory / "cost_scale.json")], ROOT)
    return {"condition": condition, "seed": seed, "seconds": time.monotonic() - started}


def _should_run_a0(conditions: list[str], seeds: list[int], force: bool, root: Path) -> bool:
    return "A0" in conditions and 9 in seeds and (
        force or not (root / "A0" / "seed9" / "a0_gate.json").is_file()
    )


def _pending_runs(conditions: list[str], seeds: list[int]) -> list[tuple[str, int]]:
    return [
        (condition, seed)
        for condition in conditions
        for seed in seeds
        if not (condition == "A0" and seed == 9)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "issue9-phase3")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    root = args.output_root
    if args.summarize_only:
        summary = summarize_only(root)
        (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return 0
    conditions = [value for value in args.conditions.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown or set(CONDITIONS.values()) - set(registered_spec_ids()):
        parser.error(f"unknown or unregistered condition: {unknown}")
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    root.mkdir(parents=True, exist_ok=True)
    failures = []
    results = []
    # A0 is deliberately isolated and must pass before any other work starts.
    if _should_run_a0(conditions, seeds, args.force, root):
        try:
            results.append(_run_one(root, "A0", 9, args.force))
            report = json.loads((root / "A0" / "seed9" / "report.json").read_text(encoding="utf-8"))
            evaluate = json.loads((root / "A0" / "seed9" / "evaluate.json").read_text(encoding="utf-8"))
            gate = _gate(report, evaluate)
            (root / "A0" / "seed9" / "a0_gate.json").write_text(json.dumps(gate, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            if not gate["passed"]:
                print(json.dumps(gate, ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)
                return 2
        except Exception as exc:
            print(f"A0 failed: {exc}", file=sys.stderr)
            return 2
    pending = _pending_runs(conditions, seeds)
    jobs = args.jobs
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(_run_one, root, condition, seed, args.force): (condition, seed) for condition, seed in pending}
        for future in as_completed(futures):
            condition, seed = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"condition": condition, "seed": seed, "error": str(exc)})
    summary = summarize_only(root)
    summary.update({"run_seconds": sorted(results, key=lambda item: (item["condition"], item["seed"])), "failed": summary["failed"] + failures})
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
