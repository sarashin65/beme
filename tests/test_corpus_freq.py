from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from beme.corpus_freq import CorpusFrequencyTable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
from build_d1_data import article_split, load_exclusion_terms  # noqa: E402


def test_table_round_trip_is_exact(tmp_path: Path) -> None:
    table = CorpusFrequencyTable.from_counts(
        {("かな", "仮名"): 3, ("かんじ", "漢字"): 2},
        {"仮名": 3, "漢字": 2},
        5,
    )
    path = tmp_path / "table.bin"
    digest = table.dump(path)
    loaded = CorpusFrequencyTable.load(path)
    assert loaded.total_tokens == 5
    assert loaded.pair_count("かな", "仮名") == 3
    assert loaded.reading_count("かな") == 3
    assert loaded.surface_count("漢字") == 2
    assert loaded.pair_count("なし", "なし") == 0
    assert digest == __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def test_dump_is_deterministic(tmp_path: Path) -> None:
    first = CorpusFrequencyTable.from_counts(
        {("b", "β"): 2, ("a", "α"): 1}, {"β": 2, "α": 1}, 3
    )
    second = CorpusFrequencyTable.from_counts(
        {("a", "α"): 1, ("b", "β"): 2}, {"α": 1, "β": 2}, 3
    )
    assert first.dump(tmp_path / "first.bin") == second.dump(tmp_path / "second.bin")
    assert (tmp_path / "first.bin").read_bytes() == (tmp_path / "second.bin").read_bytes()


def test_load_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="scripts/build_corpus_freq.py"):
        CorpusFrequencyTable.load(tmp_path / "missing.bin")


def test_truncated_table_raises(tmp_path: Path) -> None:
    path = tmp_path / "table.bin"
    CorpusFrequencyTable.from_counts({("a", "A"): 1}, {"A": 1}, 1).dump(path)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError):
        CorpusFrequencyTable.load(path)


def _article_id_for_split(split: str, seed: int = 20260822) -> str:
    for number in range(100_000):
        article_id = f"synthetic-{split}-{number}"
        if article_split(article_id, seed) == split:
            return article_id
    raise AssertionError(f"could not find {split} article id")


def _write_articles(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _run_builder(tmp_path: Path, articles: Path, *, d1_dir: Path | None = None, allow_missing: bool = True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "table.bin"
    manifest = tmp_path / "manifest.json"
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/build_corpus_freq.py"),
        "--articles-jsonl",
        str(articles),
        "--output",
        str(output),
        "--manifest",
        str(manifest),
        "--d1-dir",
        str(d1_dir or tmp_path / "d1"),
        "--coverage-pairs",
        str(tmp_path / "no-coverage-pairs.jsonl"),
        "--progress-every",
        "0",
    ]
    if allow_missing:
        command.append("--allow-missing-d1")
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, text=True, capture_output=True)
    return result, output, json.loads(manifest.read_text(encoding="utf-8"))


def test_only_train_bucket_articles_are_counted(tmp_path: Path) -> None:
    ids = {split: _article_id_for_split(split) for split in ("train", "dev", "test")}
    rows = [
        {"id": ids["train"], "text": "訓練専用語彙を数える。"},
        {"id": ids["dev"], "text": "開発専用語彙を数えない。"},
        {"id": ids["test"], "text": "試験専用語彙を数えない。"},
    ]
    articles = tmp_path / "articles.jsonl"
    _write_articles(articles, rows)
    result, output, manifest = _run_builder(tmp_path, articles)
    assert result.returncode == 0, result.stderr
    table = CorpusFrequencyTable.load(output)
    assert table.surface_count("訓練") > 0
    assert table.surface_count("開発") == 0
    assert table.surface_count("試験") == 0
    assert manifest["articles"]["dev_test_skipped"] == 2


def test_evaluation_overlap_article_is_excluded(tmp_path: Path) -> None:
    term = load_exclusion_terms()[0]
    article_id = _article_id_for_split("train")
    articles = tmp_path / "articles.jsonl"
    _write_articles(articles, [{"id": article_id, "text": f"{term} 評価重複専用語彙。"}])
    result, output, manifest = _run_builder(tmp_path, articles)
    assert result.returncode == 0, result.stderr
    assert CorpusFrequencyTable.load(output).surface_count("評価") == 0
    assert manifest["articles"]["evaluation_overlap_skipped"] == 1
    assert manifest["articles"]["train_counted"] == 0


def test_leak_audit_fails_on_dirty_d1(tmp_path: Path) -> None:
    article_id = _article_id_for_split("train")
    articles = tmp_path / "articles.jsonl"
    _write_articles(articles, [{"id": article_id, "text": "監査専用語彙。"}])
    d1_dir = tmp_path / "d1"
    d1_dir.mkdir()
    for split in ("train", "dev", "test"):
        row = {"article_id": article_id if split == "dev" else f"other-{split}", "reading": "かんさ", "positive": "監査", "span_tokens": 1}
        (d1_dir / f"{split}.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    result, _, manifest = _run_builder(tmp_path, articles, d1_dir=d1_dir, allow_missing=False)
    assert result.returncode != 0
    assert manifest["leak_audit"]["passed"] is False
    assert manifest["leak_audit"]["counted_article_ids_intersecting_d1_dev_test"] == 1


def test_missing_d1_is_not_a_silent_pass(tmp_path: Path) -> None:
    articles = tmp_path / "articles.jsonl"
    _write_articles(articles, [{"id": _article_id_for_split("train"), "text": "監査専用語彙。"}])
    d1_dir = tmp_path / "empty-d1"
    d1_dir.mkdir()
    failed, _, manifest = _run_builder(tmp_path, articles, d1_dir=d1_dir, allow_missing=False)
    assert failed.returncode != 0
    assert manifest["leak_audit"]["d1_available"] is False
    passed, _, allowed_manifest = _run_builder(tmp_path / "allowed", articles, d1_dir=d1_dir, allow_missing=True)
    assert passed.returncode == 0
    assert allowed_manifest["leak_audit"]["d1_available"] is False
    assert allowed_manifest["leak_audit"]["passed"] is False


def test_marginals_are_consistent(tmp_path: Path) -> None:
    articles = tmp_path / "articles.jsonl"
    _write_articles(articles, [{"id": _article_id_for_split("train"), "text": "限界検査。"}])
    result, output, manifest = _run_builder(tmp_path, articles)
    assert result.returncode == 0, result.stderr
    table = CorpusFrequencyTable.load(output)
    assert manifest["counts"]["marginals_consistent"] is True
    assert sum(table.reading_count(reading) for reading, _ in table._readings) == table.total_tokens
    assert sum(table.surface_count(surface) for surface, _ in table._surfaces) == table.total_tokens
