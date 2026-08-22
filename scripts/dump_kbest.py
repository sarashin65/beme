#!/usr/bin/env python3
"""Dump deterministic K-best paths and build the Phase 1 golden readings."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beme.benchmark import katakana_to_hiragana, load_ajimee_cases, load_mozc_cases
from beme.dictionary import Dictionary, is_hiragana
from beme.lattice import build_lattice, k_best_viterbi


DEFAULT_DICTIONARY = REPOSITORY_ROOT / "data" / "dictionary.tsv"
DEFAULT_EVALUATION_CASES = REPOSITORY_ROOT / "data" / "evaluation_cases.jsonl"
DEFAULT_MOZC_CASES = REPOSITORY_ROOT / "data" / "mozc_oss.tsv"
DEFAULT_AJIMEE_CASES = REPOSITORY_ROOT / "data" / "ajimee_bench_jwtd_v2_v1.json"
SYNTHETIC_CASES = (
    "きしゃのきしゃがきしゃできしゃした",
    "ゔゔゔゔゔ",
    "きしゃ123かな!",
    "あいうえお",
    "ゔゔ",
    "かか",
)


def dictionary_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def build_cases() -> list[str]:
    cases: list[str] = []
    seen: set[str] = set()

    def add(reading: str) -> None:
        if reading and reading not in seen:
            seen.add(reading)
            cases.append(reading)

    with (REPOSITORY_ROOT / "data" / "evaluation_cases.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            add(json.loads(line)["reading"])

    for case in load_mozc_cases(DEFAULT_MOZC_CASES)[:50]:
        if is_hiragana(case.reading):
            add(case.reading)

    for case in load_ajimee_cases(DEFAULT_AJIMEE_CASES)[:20]:
        normalized = katakana_to_hiragana(case.input)
        for span in re.findall(r"[ぁ-ゖー]+", normalized):
            add(span)

    for reading in SYNTHETIC_CASES:
        add(reading)
    return cases


def write_cases(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{reading}\n" for reading in build_cases()), encoding="utf-8")


def path_record(path: object) -> dict[str, object]:
    tokens = path.tokens
    return {
        "cost": path.cost,
        "surface": path.surface,
        "tokens": [[token.reading, token.surface, token.pos, token.cost] for token in tokens],
    }


def dump_kbest(dictionary_path: Path, k: int, cases_path: Path, output_path: Path) -> None:
    dictionary = Dictionary.from_tsv(dictionary_path)
    metadata = {
        "metadata": {
            "git_rev": git_revision(),
            "dictionary_sha256": dictionary_sha256(dictionary_path),
            "k": k,
            "python_version": platform.python_version(),
        }
    }
    lines = [json.dumps(metadata, ensure_ascii=False, sort_keys=True)]
    for raw_reading in cases_path.read_text(encoding="utf-8").splitlines():
        reading = raw_reading.strip()
        if not reading or reading.startswith("#"):
            continue
        lattice = build_lattice(reading, dictionary, allow_literals=True)
        paths = k_best_viterbi(lattice, beam_width=k)
        lines.append(json.dumps({"reading": reading, "paths": [path_record(path) for path in paths]}, ensure_ascii=False))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-cases", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.k <= 0:
        raise SystemExit("--k must be positive")
    if args.build_cases:
        write_cases(args.output)
        return 0
    if args.cases is None:
        raise SystemExit("--cases is required unless --build-cases is used")
    dump_kbest(args.dictionary, args.k, args.cases, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
