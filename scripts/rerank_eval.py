#!/usr/bin/env python3
"""Two-stage reranker evaluation: score once, combine many times."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
from time import perf_counter
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beme.converter import Candidate, Converter  # noqa: E402
from beme.dictionary import Dictionary  # noqa: E402
from beme.features import extractor_for  # noqa: E402
from beme.reranker import (  # noqa: E402
    ConversionContext,
    CrossEncoderReranker,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_CACHE_DIR,
    LLMLogitsReranker,
    MockContextReranker,
)
from beme.rerank_stats import (  # noqa: E402
    gold_rank_histogram,
    paired_table,
    pair_level,
    ranked_candidates_from_scores,
)
from beme.scorer import LinearScorer  # noqa: E402
from beme.training_data import TrainingExample, load_jsonl  # noqa: E402
from beme.weights import Weights  # noqa: E402


DEFAULT_VITERBI_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.94, 8.0)
DEFAULT_RERANK_K = (5, 10, 20, 64)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("must not be empty")
    return values


def _int_or_none_list(value: str) -> tuple[int | None, ...]:
    values: list[int | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        if item in {"none", "null"}:
            values.append(None)
        else:
            values.append(_positive_int(item))
    if not values:
        raise argparse.ArgumentTypeError("must not be empty")
    return tuple(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _atomic_write(path: Path, text: str, *, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=prefix, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _jsonl_text(records: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for record in records)


def _reranker(args: argparse.Namespace) -> tuple[Any, str, dict[str, Any]]:
    if args.reranker == "mock":
        return MockContextReranker(), "mock", {"model": None, "revision": None}
    if args.reranker == "cross-encoder":
        ranker = CrossEncoderReranker(
            model_name=args.cross_encoder_model,
            cache_dir=args.cross_encoder_cache_dir,
            local_files_only=args.cross_encoder_local_files_only,
            viterbi_weight=0.0,
            cross_encoder_weight=1.0,
        )
        return ranker, "cross-encoder", {
            "model": ranker.model_name,
            "revision": getattr(ranker._model, "revision", None),
        }
    ranker = LLMLogitsReranker(
        args.model_path,
        viterbi_weight=0.0,
        llm_weight=1.0,
    )
    return ranker, "llm", {"model": str(ranker.model_path), "revision": None}


def _relevance(ranker: Any, candidates: Sequence[Candidate], context: ConversionContext, kind: str) -> tuple[float, ...]:
    ranked = ranker.rerank(candidates, context)
    scores = {candidate.surface: candidate.score for candidate in ranked}
    if set(scores) != {candidate.surface for candidate in candidates}:
        raise ValueError("reranker must return every candidate exactly once")
    if kind == "mock":
        return tuple(candidate.cost - float(scores[candidate.surface]) for candidate in candidates)
    return tuple(-float(scores[candidate.surface]) for candidate in candidates)


def score_examples(
    examples: Sequence[TrainingExample],
    converter: Converter,
    ranker: Any,
    kind: str,
) -> list[dict[str, Any]]:
    """Create score records.  This is the only function that calls a ranker."""

    records: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        started = perf_counter()
        context = ConversionContext(example.left_context, example.right_context)
        result = converter.convert(example.reading, k=converter.beam_width, context=ConversionContext(), allow_literals=True)
        candidates = result.candidates
        relevance = None if context.is_empty else _relevance(ranker, candidates, context, kind)
        records.append({
            "id": example.id or str(index),
            "reading": example.reading,
            "gold": example.target_surface,
            "contrast_pair_id": example.contrast_pair_id,
            "span_tokens": example.span_tokens,
            "reading_length": len(example.reading),
            "context_empty": context.is_empty,
            "left_context_length": len(example.left_context),
            "right_context_length": len(example.right_context),
            "confidence": result.confidence,
            "candidates": [
                {"surface": candidate.surface, "cost": candidate.cost,
                 "relevance": None if relevance is None else relevance[position]}
                for position, candidate in enumerate(candidates)
            ],
            "elapsed_seconds": perf_counter() - started,
        })
    return records


def _read_scores(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    provenance: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if line_number == 1 and value.get("record_type") == "provenance":
                provenance = value
            else:
                records.append(value)
    ids = [record.get("id") for record in records]
    if any(not isinstance(identifier, str) for identifier in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: records must have unique string ids")
    return provenance, records


def _rank(order: Sequence[dict[str, Any]], gold: str) -> int | str:
    for index, candidate in enumerate(order, 1):
        if candidate["surface"] == gold:
            return index
    return "absent"


def _top1_stats(records: Sequence[dict[str, Any]], orders: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    before_records = []
    after_records = []
    changes: list[dict[str, Any]] = []
    for record in records:
        before = record["candidates"]
        after = orders[record["id"]]
        before_rank = _rank(before, record["gold"])
        after_rank = _rank(after, record["gold"])
        before_is_correct = before_rank == 1
        after_is_correct = after_rank == 1
        before_records.append({"id": record["id"], "correct": before_is_correct})
        after_records.append({"id": record["id"], "correct": after_is_correct})
        if before_is_correct != after_is_correct:
            changes.append({
                "id": record["id"], "reading": record["reading"], "gold": record["gold"],
                "before_top1": before[0]["surface"] if before else None,
                "after_top1": after[0]["surface"] if after else None,
                "gold_rank_before": before_rank, "gold_rank_after": after_rank,
            })
    table = paired_table(before_records, after_records)
    return {
        "examples": len(records),
        "correct": table["a"] + table["b"],
        "accuracy": (table["a"] + table["b"]) / len(records) if records else 0.0,
        "gold_rank": gold_rank_histogram(records),
        "after_correct": table["a"] + table["c"],
        "after_accuracy": (table["a"] + table["c"]) / len(records) if records else 0.0,
        "gold_rank_after": gold_rank_histogram([
            {"gold": record["gold"], "candidates": orders[record["id"]]} for record in records
        ]),
        "gained": sorted((item for item in changes if item["gold_rank_before"] != 1 and item["gold_rank_after"] == 1), key=lambda item: item["id"]),
        "lost": sorted((item for item in changes if item["gold_rank_before"] == 1 and item["gold_rank_after"] != 1), key=lambda item: item["id"]),
    }


def combine_records(
    records: Sequence[dict[str, Any]],
    viterbi_weight: float,
    rerank_k: int | None,
    *,
    ce_weight: float = 1.0,
    gate: float | None = None,
) -> dict[str, Any]:
    """Combine cached relevance scores without constructing or calling a ranker."""

    selected = [
        record for record in records
        if gate is None or (record.get("confidence") is not None and record["confidence"] < gate)
    ]
    orders: dict[str, Sequence[dict[str, Any]]] = {}
    for record in records:
        should_rank = not record.get("context_empty", False) and (gate is None or record in selected)
        candidates = record["candidates"]
        if not should_rank:
            candidates = [{**candidate, "relevance": None} for candidate in candidates]
        orders[record["id"]] = ranked_candidates_from_scores(
            candidates,
            viterbi_weight=viterbi_weight,
            reranker_weight=ce_weight,
            rerank_k=rerank_k,
        )
    all_stats = _top1_stats(records, orders)
    contrast = [record for record in records if record.get("contrast_pair_id") is not None]
    contrast_stats = _top1_stats(contrast, {record["id"]: orders[record["id"]] for record in contrast})

    if rerank_k is not None:
        for record in records:
            cutoff = min(rerank_k, len(record["candidates"]))
            if cutoff == 0:
                continue
            before_surfaces = {item["surface"] for item in record["candidates"][:cutoff]}
            after_surfaces = {item["surface"] for item in orders[record["id"]][:cutoff]}
            if before_surfaces != after_surfaces:
                raise ValueError("surface recall invariance violated")

    return {
        "viterbi_weight": viterbi_weight,
        "rerank_k": rerank_k,
        "gate": gate,
        "examples": len(records),
        "context_empty": sum(record.get("context_empty", False) for record in records),
        "all": all_stats,
        "contrast": contrast_stats,
        "pair": pair_level([
            {
                "contrast_pair_id": record.get("contrast_pair_id"),
                "correct": _rank(orders[record["id"]], record["gold"]) == 1,
            }
            for record in records
        ]),
        "latency": {
            "host": socket.gethostname(),
            "total_seconds": sum(float(record.get("elapsed_seconds", 0.0)) for record in records),
            "per_example_seconds": (sum(float(record.get("elapsed_seconds", 0.0)) for record in records) / len(records) if records else 0.0),
            "per_candidate_seconds": (sum(float(record.get("elapsed_seconds", 0.0)) for record in records) / sum(len(record["candidates"]) for record in records) if records and sum(len(record["candidates"]) for record in records) else 0.0),
        },
    }


def _provenance(args: argparse.Namespace, examples: Sequence[TrainingExample], reranker_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_type": "provenance",
        "data": {"path": _display_path(args.data), "examples": len(examples), "sha256": _sha256(args.data)},
        "dictionary": {"path": _display_path(args.dictionary), "sha256": _sha256(args.dictionary)},
        "weights": {"path": _display_path(args.weights), "sha256": _sha256(args.weights)},
        "reranker": reranker_info,
        "beam_width": args.beam_width,
        "torch_threads": args.torch_threads,
        "host": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }


def run_score(args: argparse.Namespace) -> int:
    if args.torch_threads is not None:
        import torch
        torch.set_num_threads(args.torch_threads)
    examples = load_jsonl(args.data)
    dictionary = Dictionary.from_tsv(args.dictionary)
    weights = Weights.load(args.weights, feature_spec=None)
    extractor = extractor_for(weights.feature_spec, dictionary.reading_conditional)
    scorer = LinearScorer(extractor, weights)
    converter = Converter(dictionary, beam_width=args.beam_width, scorer=scorer)
    ranker, kind, model_info = _reranker(args)
    records = score_examples(examples, converter, ranker, kind)
    text = _jsonl_text([_provenance(args, examples, {"kind": kind, **model_info}), *records])
    _atomic_write(args.output, text, prefix=".rerank-score-")
    return 0


def run_combine(args: argparse.Namespace) -> int:
    provenance, records = _read_scores(args.scores[0])
    if len(args.scores) > 1:
        for path in args.scores[1:]:
            other_provenance, other_records = _read_scores(path)
            if [record["id"] for record in other_records] != [record["id"] for record in records]:
                raise ValueError("score inputs must have identical id order")
            records.extend(other_records)
            provenance = {"inputs": [provenance, other_provenance]}
    grid = []
    for vw in args.viterbi_weight:
        for rerank_k in args.rerank_k:
            for gate in args.gate or (None,):
                grid.append(combine_records(records, vw, rerank_k, gate=gate))
    output = {"provenance": provenance, "grid": grid}
    text = json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    _atomic_write(args.output, text, prefix=".rerank-combine-")
    if args.json:
        print(text, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="run the expensive reranker once")
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--dictionary", type=Path, default=ROOT / "data" / "dictionary_v2.tsv")
    score.add_argument("--weights", type=Path, required=True)
    score.add_argument("--reranker", choices=("mock", "cross-encoder", "llm"), default="cross-encoder")
    score.add_argument("--beam-width", type=_positive_int, default=64)
    score.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    score.add_argument("--cross-encoder-cache-dir", type=Path, default=DEFAULT_CROSS_ENCODER_CACHE_DIR)
    score.add_argument("--cross-encoder-local-files-only", action="store_true")
    score.add_argument("--model-path", type=Path)
    score.add_argument("--torch-threads", type=_positive_int)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(func=run_score)

    combine = subparsers.add_parser("combine", help="evaluate cached scores")
    combine.add_argument("--scores", type=Path, nargs="+", required=True)
    combine.add_argument("--viterbi-weight", type=_float_list, default=DEFAULT_VITERBI_WEIGHTS)
    combine.add_argument("--rerank-k", type=_int_or_none_list, default=DEFAULT_RERANK_K)
    combine.add_argument("--gate", type=_float_list)
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("--json", action="store_true")
    combine.set_defaults(func=run_combine)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "score" and args.reranker == "llm" and args.model_path is None:
        raise SystemExit("--model-path is required for llm")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
