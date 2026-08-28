"""Tests for the F2 corpus frequency table and its leak-safe builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
SRC_ROOT = REPOSITORY_ROOT / "src"
for path in (str(SCRIPTS_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from build_d1_data import (  # noqa: E402
    article_split,
    load_exclusion_terms,
    overlaps_evaluation,
)
from beme.corpus_freq import (  # noqa: E402
    CorpusFrequencyError,
    CorpusFrequencyTable,
)

SPLIT_SEED = 20260822


def _find_article_ids_for_splits(count_per_split: int = 1) -> dict[str, str]:
    """Find synthetic article ids that hash into each split bucket.

    The bucket is decided by article_split's actual hash, so this probes ids
    instead of hardcoding hash values.
    """
    found: dict[str, str] = {}
    for i in range(100_000):
        article_id = f"cf-test-{i}"
        split = article_split(article_id, SPLIT_SEED)
        if found.get(split) is None:
            found[split] = article_id
        if all(split in found for split in ("train", "dev", "test")):
            return {name: found[name] for name in ("train", "dev", "test")}
    raise AssertionError("could not find article ids for all three splits")


def _run_builder(tmp_path: Path, articles_jsonl: Path, **extra: object) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(SCRIPTS_ROOT / "build_corpus_freq.py"),
        "--articles-jsonl",
        str(articles_jsonl),
        "--output",
        str(tmp_path / "cf.bin"),
        "--manifest",
        str(tmp_path / "cf.manifest.json"),
        "--d1-dir",
        str(tmp_path / "d1"),
        "--coverage-pairs",
        str(tmp_path / "coverage_pairs.jsonl"),
    ]
    for key, value in extra.items():
        command += [f"--{key.replace('_', '-')}", str(value)]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _write_articles(articles: list[dict[str, str]], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(article, ensure_ascii=False) for article in articles) + "\n",
        encoding="utf-8",
    )


def test_round_trip_is_exact_and_deterministic(tmp_path: Path) -> None:
    pairs = {
        ("きしゃ", "記者"): 5,
        ("きしゃ", "汽車"): 2,
        ("はし", "箸"): 7,
        ("あめ", "雨"): 1,
    }
    surfaces = {"記者": 5, "汽車": 2, "箸": 7, "雨": 1}
    total = sum(pairs.values())
    table = CorpusFrequencyTable.from_counts(pairs, surfaces, total)
    assert table.total_tokens == total
    assert table.pair_count("きしゃ", "記者") == 5
    assert table.pair_count("きしゃ", "汽車") == 2
    assert table.pair_count("きしゃ", "箸") == 0
    assert table.reading_count("きしゃ") == 7
    assert table.reading_count("はし") == 7
    assert table.reading_count("みず") == 0
    assert table.surface_count("箸") == 7
    assert table.surface_count("橋") == 0

    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    digest_a = table.dump(first)
    digest_b = table.dump(second)
    assert digest_a == digest_b
    reloaded = CorpusFrequencyTable.load(first)
    assert reloaded.readings == table.readings
    assert reloaded.surfaces == table.surfaces
    assert reloaded.pair_counts == table.pair_counts
    assert reloaded.pair_readings == table.pair_readings
    assert reloaded.pair_surfaces == table.pair_surfaces
    assert reloaded.surface_counts == table.surface_counts
    assert reloaded.total_tokens == table.total_tokens
    for reading in table.readings:
        assert reloaded.reading_count(reading) == table.reading_count(reading)
    for surface in table.surfaces:
        assert reloaded.surface_count(surface) == table.surface_count(surface)


def test_load_missing_file_raises_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(CorpusFrequencyError):
        CorpusFrequencyTable.load(tmp_path / "does-not-exist.bin")


def test_builder_counts_only_train_split_articles(tmp_path: Path) -> None:
    ids = _find_article_ids_for_splits()
    articles = [
        {
            "id": ids["train"],
            "title": "train article",
            "text": "記者は汽車を見送った。",
        },
        {
            "id": ids["dev"],
            "title": "dev article",
            "text": "箸で橋を渡った。",
        },
        {
            "id": ids["test"],
            "title": "test article",
            "text": "雨は降った。",
        },
    ]
    path = tmp_path / "articles.jsonl"
    _write_articles(articles, path)
    result = _run_builder(tmp_path, path)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "cf.manifest.json").read_text(encoding="utf-8"))
    assert manifest["articles"]["seen"] == 3
    assert manifest["articles"]["train_counted"] == 1
    assert manifest["articles"]["dev_test_skipped"] == 2
    assert manifest["leak_audit"]["passed"] is True
    assert manifest["leak_audit"]["counted_article_ids_intersecting_d1_dev_test"] == 0

    table = CorpusFrequencyTable.load(tmp_path / "cf.bin")
    # Only the train article's tokens were counted.
    assert table.pair_count("きしゃ", "記者") > 0
    assert table.pair_count("はし", "箸") == 0
    assert table.pair_count("あめ", "雨") == 0
    assert table.total_tokens == sum(table.pair_counts)


def test_builder_excludes_evaluation_overlapping_articles(tmp_path: Path) -> None:
    ids = _find_article_ids_for_splits()
    terms = load_exclusion_terms()
    assert terms, "expected exclusion terms from data/evaluation_cases.jsonl"
    term = max(terms, key=len)
    train_id = ids["train"]
    articles = [
        {"id": train_id, "title": "overlap", "text": f"本文です。{term}です。"},
        {"id": train_id, "title": "clean", "text": "記者は汽車を見送った。"},
    ]
    # Sanity: the first article must actually trip overlaps_evaluation.
    assert overlaps_evaluation(articles[0]["text"], terms)
    path = tmp_path / "articles.jsonl"
    _write_articles(articles, path)
    result = _run_builder(tmp_path, path)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "cf.manifest.json").read_text(encoding="utf-8"))
    assert manifest["articles"]["evaluation_overlap_skipped"] == 1
    table = CorpusFrequencyTable.load(tmp_path / "cf.bin")
    assert table.pair_count("きしゃ", "記者") > 0
    # The overlapped article's unique token must not have been counted.
    assert table.pair_count("みほん", "本文") == 0


def test_builder_does_not_depend_on_d1_jsonl_or_corpus_bin(tmp_path: Path) -> None:
    # The builder must work with an empty d1 dir (leak audit finds nothing to
    # intersect) and no corpus_freq_v1.bin present.
    ids = _find_article_ids_for_splits()
    articles = [
        {"id": ids["train"], "title": "t", "text": "記者は汽車を見送った。"},
    ]
    path = tmp_path / "articles.jsonl"
    _write_articles(articles, path)
    d1_dir = tmp_path / "d1"
    d1_dir.mkdir()
    result = _run_builder(tmp_path, path)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "cf.manifest.json").read_text(encoding="utf-8"))
    assert manifest["leak_audit"]["d1_dev_test_article_ids"] == 0
    assert manifest["leak_audit"]["passed"] is True
    assert manifest["coverage"].get("error", "").endswith("coverage_pairs.jsonl") or (
        manifest["coverage"]["d1_test_gold_pairs"] == 0
    )
