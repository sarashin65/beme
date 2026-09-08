"""Unit tests for the sentence-level E0 candidate extractor."""

from scripts.e0_extract_candidates import select_candidates, split_sentences
from scripts.e0_leakage_common import build_e0_leakage_index, check_e0_leakage


def test_sentence_boundary_matches_d1_rule() -> None:
    assert split_sentences("第一文。第二文！\n第三文？\r\n") == ["第一文。", "第二文！", "第三文？"]


def _candidate(article_id: str, index: int, stratum: str = "9-16") -> dict[str, object]:
    return {
        "article_id": article_id, "sentence_index": index,
        "normalized_surface": f"本文{article_id}-{index}", "selection_stratum": stratum,
    }


def test_hash_selection_is_deterministic_and_respects_article_cap() -> None:
    rows = [_candidate(f"a-{i}", 0) for i in range(6)] + [_candidate("a-0", i) for i in range(1, 4)]
    first = select_candidates(rows, target_count=5, seed=20260908, min_articles=4, max_per_article=2)
    second = select_candidates(rows, target_count=5, seed=20260908, min_articles=4, max_per_article=2)
    assert [row["normalized_surface"] for row in first] == [row["normalized_surface"] for row in second]
    assert len({row["article_id"] for row in first}) >= 4
    assert max(sum(row["article_id"] == article for row in first) for article in {row["article_id"] for row in first}) <= 2


def test_leakage_exclusion_uses_shared_s2_function(tmp_path) -> None:
    (tmp_path / "train.jsonl").write_text(
        '{"article_id":"d1-a","text":"これはリーク対象の長い本文です。"}\n', encoding="utf-8"
    )
    index = build_e0_leakage_index(tmp_path)
    findings = check_e0_leakage(
        [{"id": "new", "article_id": "new-a", "source_text": "これはリーク対象の長い本文です。"}], index
    )
    assert findings[0]["leakage_detected"]
    assert {item["type"] for item in findings[0]["overlaps"]} >= {"document", "12gram"}
