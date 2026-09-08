#!/usr/bin/env python3
"""Fetch fixed-revision Wikipedia article bodies for E0 candidate extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="wikimedia/wikipedia")
    parser.add_argument("--config", default="20231101.ja")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-rows", type=int, default=12000)
    args = parser.parse_args()

    manifest_rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    excluded = {str(row["article_id"]) for row in manifest_rows if "article_id" in row}
    from datasets import load_dataset

    rows = load_dataset(args.dataset, args.config, split="train", streaming=True, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    scanned = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            scanned += 1
            article_id = str(row.get("id", ""))
            text = row.get("text")
            if not article_id or not isinstance(text, str) or not text or article_id in excluded:
                if scanned >= args.max_rows:
                    break
                continue
            handle.write(json.dumps({"id": article_id, "title": row.get("title", ""), "text": text,
                                     "url": row.get("url", "")}, ensure_ascii=False) + "\n")
            count += 1
            if scanned >= args.max_rows:
                break
    print(json.dumps({"scanned": scanned, "written": count, "excluded_manifest_ids": len(excluded)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
