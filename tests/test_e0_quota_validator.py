from scripts.e0_extract_candidates import select_candidates
from scripts.e0_quota_validator import check_quota, quota_gap, suggest_quota_adjustment


def rows(counts):
    result = []
    index = 0
    for span, count in counts.items():
        for _ in range(count):
            result.append({"article_id": f"a-{index}", "sentence_index": 0,
                           "normalized_surface": f"s-{index}", "selection_stratum": {
                               2: "2-4", 5: "5-8", 9: "9-16", 17: "17+"}[span],
                           "span_tokens": span, "reading_length": 40})
            index += 1
    return result


def test_quota_reports_distribution_and_gates() -> None:
    report = check_quota(rows({2: 200, 5: 500, 9: 800, 17: 500}))
    assert report["passed"]
    assert report["counts"] == {"2-4": 200, "5-8": 500, "9-16": 800, "17+": 500}


def test_quota_gap_and_adjustment_identify_short_strata() -> None:
    candidates = rows({2: 200, 5: 100, 9: 800, 17: 500})
    assert quota_gap(candidates)["5-8"] > 0
    suggestion = suggest_quota_adjustment(candidates, 2000)
    assert suggestion["priority"][0] == "5-8"


def test_fill_quota_uses_span_strata() -> None:
    candidates = rows({2: 20, 5: 30, 9: 40, 17: 30})
    selected = select_candidates(candidates, 20, 20260908, 20, fill_quota=True)
    counts = check_quota(selected)["counts"]
    assert sum(counts.values()) == 20
    assert counts["9-16"] >= 8
