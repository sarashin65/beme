"""Tests for the shared E0 leakage index."""

from pathlib import Path
import json

from scripts.e0_leakage_common import (
    build_e0_leakage_index,
    check_e0_leakage,
    document_ngrams,
    normalize_text,
)


def test_nfkc_and_whitespace_normalization() -> None:
    assert normalize_text(" ＡＢＣ\u3000\n１２３ ") == "ABC123"


def test_12gram_fingerprint_is_normalized_and_contiguous() -> None:
    assert document_ngrams("Ａ" * 12 + "\u3000") == {"A" * 12}
    assert document_ngrams("short") == set()


def test_index_and_known_leak_are_detected(tmp_path: Path) -> None:
    for split, article_id, text in (
        ("train", "article-train", "これは重複する記事本文です。"),
        ("dev", "article-dev", "開発用の別記事本文です。"),
        ("test", "article-test", "評価用の別記事本文です。"),
    ):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps({"article_id": article_id, "text": text}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    index = build_e0_leakage_index(tmp_path)
    findings = check_e0_leakage(
        [{"case_id": "case-1", "article_id": "new", "text": "これは重複する記事本文です。"},
         {"case_id": "case-2", "article_id": "article-test", "text": "新しい候補文です。"}],
        index,
    )
    assert {item["type"] for item in findings[0]["overlaps"]} >= {"document", "12gram"}
    assert any(item["type"] == "article_id" and item["split"] == "test" for item in findings[1]["overlaps"])

