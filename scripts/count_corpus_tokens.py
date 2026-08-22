#!/usr/bin/env python3
"""Count a corpus with a pinned Hugging Face tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPOSITORY_ROOT / "data" / "corpus.txt"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_tokens(corpus: Path, model: str, revision: str | None, batch_size: int) -> dict[str, object]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)
    token_count = 0
    nonempty_lines = 0
    batch: list[str] = []
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.rstrip("\n")
            if not text:
                continue
            nonempty_lines += 1
            batch.append(text)
            if len(batch) == batch_size:
                token_count += sum(
                    len(ids) for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]
                )
                batch.clear()
        if batch:
            token_count += sum(
                len(ids) for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]
            )
    return {
        "corpus": str(corpus),
        "corpus_sha256": file_sha256(corpus),
        "utf8_bytes": corpus.stat().st_size,
        "nonempty_lines": nonempty_lines,
        "tokenizer": model,
        "requested_revision": revision,
        "resolved_revision": tokenizer.init_kwargs.get("_commit_hash") or revision,
        "native_tokens": token_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = count_tokens(args.corpus, args.model, args.revision, args.batch_size)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
