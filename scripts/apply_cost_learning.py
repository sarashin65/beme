#!/usr/bin/env python3
"""Learn word frequencies from a corpus and apply them to the BeME TSV."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beme.ipadic import katakana_to_hiragana, map_ipadic_pos  # noqa: E402


DEFAULT_CORPUS = REPOSITORY_ROOT / "data" / "corpus.txt"
DEFAULT_INPUT = REPOSITORY_ROOT / "data" / "dictionary.tsv"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "dictionary_learned.tsv"
DEFAULT_TOP_N = 5_000
DEFAULT_ALPHA = 0.1
DEFAULT_MIN_COST = 0.1
_TAGGER = None


def _get_tagger():
    """Create one IPADIC tagger per process, rather than per input line."""
    global _TAGGER
    if _TAGGER is None:
        try:
            from fugashi import GenericTagger
            import ipadic
        except ImportError as exc:
            raise RuntimeError("fugashi and ipadic are required; run 'uv sync --group dev' first.") from exc
        # fugashi bundles the MeCab runtime, but an explicit null rc file is
        # needed on systems without a global mecabrc.
        _TAGGER = GenericTagger(f"-r /dev/null -d {ipadic.DICDIR}")
    return _TAGGER


def _nodes(text: str):
    tagger = _get_tagger()
    yield from tagger(text)


def count_chunk(text: str) -> Counter[tuple[str, str, str]]:
    """Count ``(hiragana reading, surface, BeME POS)`` tuples in *text*."""
    counts: Counter[tuple[str, str, str]] = Counter()
    for node in _nodes(text):
        feature = node.feature
        if isinstance(feature, str):
            fields = feature.split(",")
        else:
            fields = tuple(feature)
        if len(fields) < 7:
            continue
        reading = katakana_to_hiragana(fields[6])
        if not reading or not all("\u3041" <= char <= "\u3096" or char == "ー" for char in reading):
            continue
        pos = map_ipadic_pos(fields[0], fields[1], fields[2])
        counts[(reading, node.surface, pos)] += 1
    return counts


def _chunks(handle: Iterable[str], chunk_lines: int) -> Iterator[str]:
    lines: list[str] = []
    for line in handle:
        lines.append(line)
        if len(lines) >= chunk_lines:
            yield "".join(lines)
            lines.clear()
    if lines:
        yield "".join(lines)


def collect_frequencies(
    corpus: str | Path,
    *,
    workers: int | None = None,
    chunk_lines: int = 1_000,
) -> Counter[tuple[str, str, str]]:
    """Parse *corpus* in parallel and return deterministic aggregate counts."""
    if chunk_lines <= 0:
        raise ValueError("chunk_lines must be greater than zero")
    worker_count = workers or (os.cpu_count() or 1)
    if worker_count <= 0:
        raise ValueError("workers must be greater than zero")
    with Path(corpus).open("r", encoding="utf-8") as handle:
        chunks = _chunks(handle, chunk_lines)
        accumulator: Counter[tuple[str, str, str]] = Counter()
        if worker_count == 1:
            for chunk_counts in map(count_chunk, chunks):
                accumulator.update(chunk_counts)
            return accumulator
        with mp.Pool(worker_count) as pool:
            for chunk_counts in pool.imap(count_chunk, chunks):
                accumulator.update(chunk_counts)
        return accumulator


def select_vocabulary(
    frequencies: Counter[tuple[str, str, str]], top_n: int
) -> dict[tuple[str, str, str], int]:
    """Return the top entries, with ties ordered by their tuple value."""
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")
    return dict(sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:top_n])


def apply_costs(
    input_dictionary: str | Path,
    output_dictionary: str | Path,
    frequencies: Counter[tuple[str, str, str]],
    *,
    top_n: int = DEFAULT_TOP_N,
    alpha: float = DEFAULT_ALPHA,
    min_cost: float = DEFAULT_MIN_COST,
) -> tuple[int, int]:
    """Read *input_dictionary*, apply cost learning, and atomically write
    *output_dictionary* (which may be the same path as the input). Returns
    (matched, changed).
    """
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite non-negative number")
    if not math.isfinite(min_cost) or min_cost < 0:
        raise ValueError("min_cost must be a finite non-negative number")
    selected = select_vocabulary(frequencies, top_n)
    source = Path(input_dictionary)
    destination = Path(output_dictionary)
    rows: list[str] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows.extend(line.rstrip("\r\n") for line in handle)
    if not rows or rows[0] != "reading\tsurface\tpos\tcost":
        raise ValueError(f"{source}: expected reading\\tsurface\\tpos\\tcost header")

    matched = changed = 0
    output_rows: list[str] = [rows[0]]
    for raw in rows[1:]:
        if not raw:
            output_rows.append("")
            continue
        fields = raw.split("\t")
        if len(fields) != 4:
            raise ValueError(f"{source}: malformed dictionary row")
        frequency = selected.get((fields[0], fields[1], fields[2]))
        if frequency is not None:
            old_cost = float(fields[3])
            new_cost = max(min_cost, old_cost - alpha * math.log10(frequency))
            if new_cost < old_cost:
                fields[3] = f"{new_cost:.3f}"
                changed += 1
            matched += 1
        output_rows.append("\t".join(fields))

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=destination.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write("\n".join(output_rows))
            handle.write("\n")
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return matched, changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply corpus frequency cost learning to BeME dictionary")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="input dictionary path to read costs from (default: data/dictionary.tsv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "output dictionary path to write learned costs to (default: "
            "data/dictionary_learned.tsv; keeps --input unchanged unless the "
            "same path is passed for both)"
        ),
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--min-cost", type=float, default=DEFAULT_MIN_COST)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunk-lines", type=int, default=1_000)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Validate the input dictionary before running corpus parsing (which can
    # take minutes on the full 100MB corpus) so a bad --input path fails fast
    # instead of after all that work is discarded.
    if not args.input.is_file():
        print(f"error: --input dictionary not found: {args.input}", file=sys.stderr)
        return 1

    # Create the output directory up front too, so a missing directory is
    # reported immediately rather than after corpus parsing completes.
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frequencies = collect_frequencies(args.corpus, workers=args.workers, chunk_lines=args.chunk_lines)
    matched, changed = apply_costs(
        args.input,
        args.output,
        frequencies,
        top_n=args.top_n,
        alpha=args.alpha,
        min_cost=args.min_cost,
    )
    result = {
        "corpus": str(args.corpus),
        "input": str(args.input),
        "output": str(args.output),
        "unique_tokens": len(frequencies),
        "total_tokens": sum(frequencies.values()),
        "top_n": args.top_n,
        "matched_entries": matched,
        "changed_entries": changed,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "Updated {changed_entries} of {matched_entries} matched entries "
            "({unique_tokens} unique tokens) from {input} into {output}.".format(**result)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
