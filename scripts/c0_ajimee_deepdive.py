#!/usr/bin/env python3
"""Re-run AJIMEE oracle decomposition with C0 (A8+corpus) vs A8 baseline."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/sarashin/work/beme-9p4")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beme.benchmark import load_ajimee_cases, edit_distance, katakana_to_hiragana
from beme.converter import Converter
from beme.dictionary import Dictionary
from beme.features import extractor_for
from beme.lattice import build_lattice
from beme.oracle import constrained_best_path
from beme.scorer import LinearScorer
from beme.weights import Weights

# ── config ──────────────────────────────────────────────────────────
DICT = ROOT / "data" / "dictionary_v2.tsv"
BENCHMARK = ROOT / "data" / "ajimee_bench_jwtd_v2_v1.json"
A8_WEIGHTS = ROOT / "artifacts" / "issue19-f2" / "A8" / "seed9" / "weights.json"
C0_WEIGHTS = ROOT / "artifacts" / "issue19-f2" / "C0" / "seed9" / "weights.json"
OUTPUT = ROOT / "docs" / "c0-ajimee-deepdive-2026-09-04.json"
BEAM = 64
TOP_K = 5

# ── load ────────────────────────────────────────────────────────────
print("Loading dictionary...", flush=True)
dictionary = Dictionary.from_tsv(DICT)

print("Loading weights...", flush=True)
a8_weights = Weights.load(A8_WEIGHTS, feature_spec=None)
c0_weights = Weights.load(C0_WEIGHTS, feature_spec=None)

a8_ext = extractor_for(a8_weights.feature_spec, dictionary.reading_conditional)
c0_ext = extractor_for(c0_weights.feature_spec, dictionary.reading_conditional)
a8_scorer = LinearScorer(a8_ext, a8_weights)
c0_scorer = LinearScorer(c0_ext, c0_weights)
a8_converter = Converter(dictionary, beam_width=BEAM, scorer=a8_scorer)
c0_converter = Converter(dictionary, beam_width=BEAM, scorer=c0_scorer)

print("Loading AJIMEE cases...", flush=True)
cases = load_ajimee_cases(BENCHMARK)
print(f"  {len(cases)} cases", flush=True)

# ── classify errors ────────────────────────────────────────────────
def classify_error(case, converter, scorer, label: str) -> dict:
    """Classify a single AJIMEE conversion error using oracle DP."""
    reading = katakana_to_hiragana(case.input)
    references = case.references
    # Use the first reference as primary gold for oracle analysis
    gold = references[0]
    
    # Use convert_sentence for proper handling with context
    from beme.benchmark import convert_sentence
    result = convert_sentence(converter, case.input, k=TOP_K, cascade_threshold=2.0, preceding_context=case.context)
    top1 = result[0].surface if result else None
    correct = top1 in references if top1 else False
    
    record = {
        "id": case.id,
        "reading": reading,
        "gold": gold,
        "all_references": list(references),
        "top1": top1,
        "correct": correct,
        "gold_rank": None,
        "error_type": None,
        "edit_distance": min(edit_distance(ref, top1) for ref in references) if top1 else min(len(ref) for ref in references),
        "oracle_cost": None,
        "hyp_cost": None,
        "gold_in_lattice": False,
        "segmentation_match": None,
    }
    
    # Find gold rank in top-k
    for i, cand in enumerate(result, 1):
        if cand.surface in references:
            record["gold_rank"] = i
            break
    
    # Build lattice and find constrained gold path
    dictionary = converter.dictionary
    lattice = build_lattice(reading, dictionary, allow_literals=True)
    gold_path = constrained_best_path(lattice, gold, converter.scorer)
    
    if gold_path is not None:
        record["gold_in_lattice"] = True
        record["oracle_cost"] = gold_path.cost
        record["hyp_cost"] = result[0].cost if result else None
        
        if not correct and result and top1 is not None:
            hyp_path = constrained_best_path(lattice, top1, converter.scorer)
            if hyp_path is not None:
                gold_tokens = list(gold_path.tokens)
                hyp_tokens = list(hyp_path.tokens)
                gold_boundaries = []
                pos = 0
                for t in gold_tokens:
                    gold_boundaries.append(pos)
                    pos += len(t.reading)
                hyp_boundaries = []
                pos = 0
                for t in hyp_tokens:
                    hyp_boundaries.append(pos)
                    pos += len(t.reading)
                
                same_boundaries = gold_boundaries == hyp_boundaries
                record["segmentation_match"] = same_boundaries
                
                if same_boundaries:
                    record["error_type"] = "homophone"
                else:
                    record["error_type"] = "segmentation"
    else:
        record["error_type"] = "coverage"
    
    return record


print("Running A8 baseline evaluation...", flush=True)
a8_records = []
for i, case in enumerate(cases):
    if (i + 1) % 50 == 0:
        print(f"  A8: {i+1}/{len(cases)}", flush=True)
    rec = classify_error(case, a8_converter, a8_scorer, "A8")
    a8_records.append(rec)

print("Running C0 evaluation...", flush=True)
c0_records = []
for i, case in enumerate(cases):
    if (i + 1) % 50 == 0:
        print(f"  C0: {i+1}/{len(cases)}", flush=True)
    rec = classify_error(case, c0_converter, c0_scorer, "C0")
    c0_records.append(rec)

# ── aggregate ───────────────────────────────────────────────────────
def summarize(records, label: str) -> dict:
    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    errors = [r for r in records if not r["correct"]]
    
    by_type = Counter(r["error_type"] for r in errors)
    by_type_cer = defaultdict(float)
    for r in errors:
        by_type_cer[r["error_type"]] += r["edit_distance"]
    
    gold_in_lattice = sum(1 for r in records if r["gold_in_lattice"])
    
    return {
        "label": label,
        "total": total,
        "correct": correct,
        "errors": total - correct,
        "error_rate": (total - correct) / total,
        "mean_cer": sum(r["edit_distance"] for r in records) / total,
        "exact_match": correct / total,
        "gold_in_lattice": gold_in_lattice,
        "gold_in_lattice_pct": gold_in_lattice / total,
        "error_types": {
            et: {"count": c, "pct": c / len(errors) * 100 if errors else 0, "cer_contrib": by_type_cer[et]}
            for et, c in sorted(by_type.most_common())
        },
        "error_details": [
            {"id": r["id"], "reading": r["reading"], "gold": r["gold"], "top1": r["top1"],
             "error_type": r["error_type"], "edit_distance": r["edit_distance"],
             "gold_in_lattice": r["gold_in_lattice"], "segmentation_match": r["segmentation_match"],
             "gold_rank": r["gold_rank"]}
            for r in errors
        ],
    }

a8_summary = summarize(a8_records, "A8")
c0_summary = summarize(c0_records, "C0")

# ── comparison ──────────────────────────────────────────────────────
# Fixed/fixed/broken/still_wrong breakdown
comparison = []
for a8r, c0r in zip(a8_records, c0_records):
    comparison.append({
        "id": a8r["id"],
        "reading": a8r["reading"],
        "gold": a8r["gold"],
        "a8_top1": a8r["top1"],
        "c0_top1": c0r["top1"],
        "a8_correct": a8r["correct"],
        "c0_correct": c0r["correct"],
        "a8_error_type": a8r["error_type"],
        "c0_error_type": c0r["error_type"],
        "a8_edit_distance": a8r["edit_distance"],
        "c0_edit_distance": c0r["edit_distance"],
        "status": (
            "both_correct" if a8r["correct"] and c0r["correct"]
            else "fixed" if not a8r["correct"] and c0r["correct"]
            else "regressed" if a8r["correct"] and not c0r["correct"]
            else "still_wrong"
        ),
    })

# ── output ──────────────────────────────────────────────────────────
document = {
    "config": {
        "dictionary": str(DICT),
        "benchmark": str(BENCHMARK),
        "a8_weights": str(A8_WEIGHTS),
        "c0_weights": str(C0_WEIGHTS),
        "beam_width": BEAM,
        "top_k": TOP_K,
    },
    "a8": a8_summary,
    "c0": c0_summary,
    "comparison": {
        "total": len(comparison),
        "both_correct": sum(1 for c in comparison if c["status"] == "both_correct"),
        "fixed": sum(1 for c in comparison if c["status"] == "fixed"),
        "regressed": sum(1 for c in comparison if c["status"] == "regressed"),
        "still_wrong": sum(1 for c in comparison if c["status"] == "still_wrong"),
        "details": comparison,
    },
    "old_deepdive_reference": {
        "date": "2026-08-21",
        "commit": "24a422b",
        "baseline": "dictionary.tsv (349,043 entries)",
        "segmentation_errors": 101,
        "homophone_errors": 60,
        "coverage_errors": 25,
        "exact_match": 12,
        "search_errors": 2,
    },
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(document, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"Written to {OUTPUT}")
print(f"{'='*60}")
print(f"\nA8: {a8_summary['correct']}/{a8_summary['total']} correct, "
      f"CER={a8_summary['mean_cer']:.4f}, exact={a8_summary['exact_match']:.3f}")
print(f"C0: {c0_summary['correct']}/{c0_summary['total']} correct, "
      f"CER={c0_summary['mean_cer']:.4f}, exact={c0_summary['exact_match']:.3f}")
print(f"\nFixed:   {document['comparison']['fixed']}")
print(f"Regressed: {document['comparison']['regressed']}")
print(f"Still wrong: {document['comparison']['still_wrong']}")
print(f"\nA8 error breakdown:")
for et, info in sorted(a8_summary['error_types'].items()):
    print(f"  {et}: {info['count']} ({info['pct']:.1f}%) CER={info['cer_contrib']:.1f}")
print(f"\nC0 error breakdown:")
for et, info in sorted(c0_summary['error_types'].items()):
    print(f"  {et}: {info['count']} ({info['pct']:.1f}%) CER={info['cer_contrib']:.1f}")