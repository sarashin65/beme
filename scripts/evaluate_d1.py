#!/usr/bin/env python3
"""Evaluate Qwen2.5 D0/D1 conditional candidate likelihood and BeME suites."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beme.benchmark import (  # noqa: E402
    evaluate_ajimee,
    evaluate_mozc,
    load_ajimee_cases,
    load_mozc_cases,
)
from beme.converter import Candidate, ConversionResult, Converter  # noqa: E402
from beme.evaluation import evaluate, load_cases  # noqa: E402
from beme.reranker import ConversionContext, Reranker  # noqa: E402
from train_d1_qlora import prompt_text  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
DEFAULT_DICTIONARY = REPOSITORY_ROOT / "data" / "dictionary.tsv"
DEFAULT_DEV = REPOSITORY_ROOT / "data" / "d1" / "dev.jsonl"
DEFAULT_TEST = REPOSITORY_ROOT / "data" / "d1" / "test.jsonl"


@dataclass(frozen=True)
class CandidateScore:
    surface: str
    sum_logprob: float
    mean_logprob: float
    token_count: int


class ConditionalScorer:
    def __init__(
        self,
        model_name: str,
        model_revision: str,
        adapter: Path | None,
        dtype_name: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        compute_dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=model_revision,
            quantization_config=quantization,
            dtype=compute_dtype,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        model.config.use_cache = True
        self.model = model
        self.device = next(model.parameters()).device
        self.scored_candidates = 0
        self.elapsed_seconds = 0.0

    def _encode(self, left: str, reading: str, right: str, candidate: str) -> tuple[list[int], int]:
        eos = self.tokenizer.eos_token_id
        answer_ids = self.tokenizer.encode(candidate, add_special_tokens=False) + [eos]
        prompt_ids = self.tokenizer.encode(prompt_text(left, reading, right), add_special_tokens=False)
        while len(prompt_ids) + len(answer_ids) > self.max_length and (left or right):
            if len(left) >= len(right) and left:
                left = left[1:]
            elif right:
                right = right[:-1]
            prompt_ids = self.tokenizer.encode(prompt_text(left, reading, right), add_special_tokens=False)
        if len(prompt_ids) + len(answer_ids) > self.max_length:
            raise ValueError(f"candidate cannot fit max length {self.max_length}: {candidate!r}")
        return prompt_ids + answer_ids, len(prompt_ids)

    def score(self, left: str, reading: str, right: str, candidates: Sequence[str]) -> list[CandidateScore]:
        torch = self.torch
        encoded = [self._encode(left, reading, right, candidate) for candidate in candidates]
        scores: list[CandidateScore] = []
        started = time.perf_counter()
        with torch.inference_mode():
            for offset in range(0, len(encoded), self.batch_size):
                chunk = encoded[offset : offset + self.batch_size]
                width = max(len(ids) for ids, _ in chunk)
                input_ids, masks = [], []
                for ids, _ in chunk:
                    padding = width - len(ids)
                    input_ids.append(ids + [self.tokenizer.pad_token_id] * padding)
                    masks.append([1] * len(ids) + [0] * padding)
                input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                mask_tensor = torch.tensor(masks, dtype=torch.long, device=self.device)
                logits = self.model(input_ids=input_tensor, attention_mask=mask_tensor).logits
                for local_index, (ids, answer_start) in enumerate(chunk):
                    target = input_tensor[local_index, answer_start : len(ids)]
                    prediction = logits[local_index, answer_start - 1 : len(ids) - 1]
                    token_logprobs = torch.log_softmax(prediction.float(), dim=-1).gather(
                        1, target.unsqueeze(1)
                    ).squeeze(1)
                    total = float(token_logprobs.sum().item())
                    count = int(target.numel())
                    scores.append(CandidateScore(candidates[offset + local_index], total, total / count, count))
        self.elapsed_seconds += time.perf_counter() - started
        self.scored_candidates += len(candidates)
        return scores


class ConditionalReranker:
    """D1 conditional-likelihood implementation of the BeME Reranker protocol."""

    def __init__(self, scorer: ConditionalScorer, score_mode: str) -> None:
        if score_mode not in {"sum", "mean"}:
            raise ValueError("score_mode must be 'sum' or 'mean'")
        self.scorer = scorer
        self.score_mode = score_mode

    @staticmethod
    def _reading(candidates: Sequence[Candidate]) -> str:
        readings = {"".join(token.reading for token in candidate.tokens) for candidate in candidates}
        if len(readings) != 1:
            raise ValueError("reranker candidates must share one reading")
        reading = next(iter(readings))
        if not reading:
            raise ValueError("reranker candidates must have a non-empty reading")
        return reading

    def rerank(
        self,
        candidates: Sequence[Candidate],
        context: ConversionContext,
    ) -> tuple[Candidate, ...]:
        if not candidates:
            return ()
        reading = self._reading(candidates)
        candidate_scores = self.scorer.score(
            context.preceding,
            reading,
            context.following,
            [candidate.surface for candidate in candidates],
        )
        by_surface = {score.surface: score for score in candidate_scores}
        scored = tuple(
            replace(
                candidate,
                score=-getattr(by_surface[candidate.surface], f"{self.score_mode}_logprob"),
            )
            for candidate in candidates
        )
        return tuple(sorted(scored, key=lambda item: (item.score, item.cost, item.surface)))


class ConditionalConverter:
    """Evaluation adapter that invokes a D1 Reranker even without context."""

    def __init__(self, base: Converter, reranker: Reranker, rerank_k: int) -> None:
        self.base = base
        self.reranker = reranker
        self.rerank_k = rerank_k

    def convert(
        self,
        reading: str,
        k: int = 5,
        cascade_threshold: float = 2.0,
        context: ConversionContext | None = None,
        *,
        allow_literals: bool = False,
    ) -> ConversionResult:
        context = context or ConversionContext()
        base_result = self.base.convert(
            reading,
            k=self.rerank_k,
            cascade_threshold=cascade_threshold,
            context=context,
            allow_literals=allow_literals,
        )
        ranked = self.reranker.rerank(base_result.candidates, context)[:k]
        return replace(base_result, candidates=ranked, reranked=True)


def load_groups(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def rank_metrics(ranks: Sequence[int]) -> dict[str, float]:
    if not ranks:
        return {"conditional_top1": 0.0, "mrr": 0.0, "ndcg": 0.0}
    return {
        "conditional_top1": sum(rank == 1 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "ndcg": sum(1.0 / math.log2(rank + 1) for rank in ranks) / len(ranks),
    }



def paired_bootstrap_ci(
    improvements: int,
    regressions: int,
    total: int,
    *,
    resamples: int = 10_000,
    seed: int = 20260822,
) -> dict[str, float | int]:
    """Percentile CI for the paired Top-1 difference in accuracy points."""

    differences = (1,) * improvements + (-1,) * regressions
    differences += (0,) * (total - improvements - regressions)
    rng = random.Random(seed)
    samples = sorted(
        sum(differences[rng.randrange(total)] for _ in range(total)) / total
        for _ in range(resamples)
    )
    return {
        "resamples": resamples,
        "seed": seed,
        "lower": samples[int(resamples * 0.025)],
        "upper": samples[int(resamples * 0.975)],
    }

def evaluate_groups(
    groups: Sequence[dict[str, Any]],
    scorer: ConditionalScorer,
    context_mode: str,
    details_handle: Any,
) -> dict[str, Any]:
    ranks: dict[str, list[int]] = {"viterbi": [], "sum": [], "mean": []}
    predictions: dict[str, dict[str, bool]] = {"viterbi": {}, "sum": {}, "mean": {}}
    strata: dict[str, dict[str, dict[str, list[int]]]] = {
        "sum": defaultdict(lambda: defaultdict(list)),
        "mean": defaultdict(lambda: defaultdict(list)),
    }
    for index, group in enumerate(groups, 1):
        candidates = [str(value) for value in group["candidates"]]
        costs = [float(value) for value in group["viterbi_costs"]]
        positive = str(group["positive"])
        left = str(group["left_context"])
        right = str(group["right_context"]) if context_mode == "left+right" else ""
        candidate_scores = scorer.score(left, str(group["reading"]), right, candidates)
        score_by_surface = {score.surface: score for score in candidate_scores}
        orderings = {
            "viterbi": sorted(candidates, key=lambda value: (costs[candidates.index(value)], value)),
            "sum": sorted(candidates, key=lambda value: (-score_by_surface[value].sum_logprob, value)),
            "mean": sorted(candidates, key=lambda value: (-score_by_surface[value].mean_logprob, value)),
        }
        for mode, ordering in orderings.items():
            rank = ordering.index(positive) + 1
            ranks[mode].append(rank)
            predictions[mode][str(group["id"])] = rank == 1
            if mode != "viterbi":
                buckets = group.get("frequency_bucket", {})
                positive_bucket = str(buckets.get("positive", "unknown"))
                hard_negative_bucket = str(buckets.get("hard_negative", "unknown"))
                strata[mode]["positive_frequency_bucket"][positive_bucket].append(rank)
                strata[mode]["hard_negative_frequency_bucket"][hard_negative_bucket].append(rank)
                ambiguity = "2-5" if len(candidates) <= 5 else "6-16" if len(candidates) <= 16 else "17+"
                strata[mode]["ambiguity"][ambiguity].append(rank)
                strata[mode]["span_tokens"][str(group.get("span_tokens", "unknown"))].append(rank)
                context_length = len(left) + len(right)
                context_bucket = "0-31" if context_length < 32 else "32-63" if context_length < 64 else "64+"
                strata[mode]["context_length"][context_bucket].append(rank)
        if details_handle is not None:
            details_handle.write(
                json.dumps(
                    {
                        "id": group["id"],
                        "context_mode": context_mode,
                        "gold": positive,
                        "viterbi_rank": orderings["viterbi"].index(positive) + 1,
                        "sum_rank": orderings["sum"].index(positive) + 1,
                        "mean_rank": orderings["mean"].index(positive) + 1,
                        "candidates": [
                            {
                                "surface": candidate,
                                "viterbi_cost": costs[position],
                                "sum_logprob": score_by_surface[candidate].sum_logprob,
                                "mean_logprob": score_by_surface[candidate].mean_logprob,
                                "answer_tokens_including_eos": score_by_surface[candidate].token_count,
                            }
                            for position, candidate in enumerate(candidates)
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        if index % 100 == 0:
            print(f"conditional progress mode={context_mode} groups={index}/{len(groups)}", file=sys.stderr, flush=True)

    report: dict[str, Any] = {mode: rank_metrics(values) for mode, values in ranks.items()}
    pair_members: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        pair_id = group.get("contrast_pair_id")
        if pair_id:
            pair_members[str(pair_id)].append(str(group["id"]))
    valid_pairs = [members for members in pair_members.values() if len(members) == 2]
    contrast_item_ids = [item_id for pair in valid_pairs for item_id in pair]
    viterbi_correct = predictions["viterbi"]
    for mode in ("viterbi", "sum", "mean"):
        model_correct = predictions[mode]
        report[mode]["contrast_item_count"] = len(contrast_item_ids)
        report[mode]["contrast_item_top1"] = (
            sum(model_correct[item_id] for item_id in contrast_item_ids) / len(contrast_item_ids)
            if contrast_item_ids
            else None
        )
        report[mode]["contrast_pair_count"] = len(valid_pairs)
        report[mode]["contrast_pair_both_correct"] = (
            sum(all(model_correct[item_id] for item_id in pair) for pair in valid_pairs) / len(valid_pairs)
            if valid_pairs
            else None
        )
        if mode == "viterbi":
            continue
        improvements = sum(
            not viterbi_correct[item_id] and correct for item_id, correct in model_correct.items()
        )
        regressions = sum(
            viterbi_correct[item_id] and not correct for item_id, correct in model_correct.items()
        )
        report[mode]["top1_delta_vs_viterbi"] = (improvements - regressions) / len(groups)
        report[mode]["paired_top1_counts"] = {
            "improvements": improvements,
            "regressions": regressions,
            "unchanged": len(groups) - improvements - regressions,
        }
        report[mode]["recoverable_error_recovery"] = improvements / max(
            1, sum(not value for value in viterbi_correct.values())
        )
        report[mode]["baseline_correct_regression"] = regressions / max(
            1, sum(viterbi_correct.values())
        )
        report[mode]["strata"] = {
            dimension: {bucket: rank_metrics(bucket_ranks) for bucket, bucket_ranks in buckets.items()}
            for dimension, buckets in strata[mode].items()
        }
    report["baseline"] = "mock/existing Viterbi order"
    report["group_count"] = len(groups)
    report["oracle_recall"] = 1.0
    return report


def suite_report(converter: Any) -> dict[str, Any]:
    handmade = evaluate(converter, load_cases(REPOSITORY_ROOT / "data" / "evaluation_cases.jsonl"), k=5)
    handmade["suite"] = "handmade"
    return {
        "handmade": handmade,
        "ajimee": evaluate_ajimee(
            converter,
            load_ajimee_cases(REPOSITORY_ROOT / "data" / "ajimee_bench_jwtd_v2_v1.json"),
            k=5,
        ),
        "mozc": evaluate_mozc(
            converter,
            load_mozc_cases(REPOSITORY_ROOT / "data" / "mozc_oss.tsv"),
            k=5,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-details", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--suite-max-length", type=int, default=1024)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--rerank-k", type=int, default=64)
    suite_mode = parser.add_mutually_exclusive_group()
    suite_mode.add_argument("--skip-suites", action="store_true")
    suite_mode.add_argument("--only-suites", action="store_true")
    parser.add_argument("--score-mode", choices=("sum", "mean"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.only_suites and args.score_mode is None:
        parser.error("--only-suites requires --score-mode selected on dev")
    if not args.only_suites and args.candidate_details is None:
        parser.error("--candidate-details is required unless --only-suites is used")
    scorer = ConditionalScorer(
        args.model,
        args.model_revision,
        args.adapter,
        args.dtype,
        args.batch_size,
        args.max_length,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "label": args.label,
        "model": args.model,
        "model_revision": args.model_revision,
        "dtype": args.dtype,
        "adapter": str(args.adapter) if args.adapter else None,
    }
    if args.only_suites:
        selected_score_mode = args.score_mode
        report["selected_score_mode_from_dev"] = selected_score_mode
    else:
        dev_groups, test_groups = load_groups(args.dev), load_groups(args.test)
        args.candidate_details.parent.mkdir(parents=True, exist_ok=True)
        with args.candidate_details.open("w", encoding="utf-8", newline="\n") as details:
            for context_mode in ("left+right", "left-only"):
                dev_report = evaluate_groups(dev_groups, scorer, context_mode, details)
                selected_mode = max(
                    ("sum", "mean"),
                    key=lambda mode: (dev_report[mode]["conditional_top1"], mode == "sum"),
                )
                test_report = evaluate_groups(test_groups, scorer, context_mode, details)
                selected_test = test_report[selected_mode]
                paired_counts = selected_test["paired_top1_counts"]
                selected_test["paired_bootstrap_95_ci"] = paired_bootstrap_ci(
                    paired_counts["improvements"],
                    paired_counts["regressions"],
                    test_report["group_count"],
                )
                report[context_mode] = {
                    "selected_score_mode_from_dev": selected_mode,
                    "dev": dev_report,
                    "test": test_report,
                    "selected_test": selected_test,
                }
        selected_score_mode = report["left+right"]["selected_score_mode_from_dev"]
    scorer.max_length = args.suite_max_length
    base_converter = Converter.from_tsv(args.dictionary, beam_width=args.rerank_k)
    if not args.skip_suites:
        report["suites"] = {
            "viterbi": suite_report(base_converter),
            "model": suite_report(
                ConditionalConverter(
                    base_converter,
                    ConditionalReranker(scorer, selected_score_mode),
                    args.rerank_k,
                )
            ),
        }
    report["inference"] = {
        "scored_candidates": scorer.scored_candidates,
        "elapsed_seconds": scorer.elapsed_seconds,
        "candidate_latency_ms": 1000 * scorer.elapsed_seconds / max(1, scorer.scored_candidates),
        "peak_memory_allocated_bytes": scorer.torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": scorer.torch.cuda.max_memory_reserved(),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **report["inference"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
