from beme.dictionary import Dictionary, DictionaryEntry
from scripts.e0_gold_validator import enrich_candidate, validate_gold_paths


def _candidate(tokens):
    return {"input": "かな", "canonical_reference": "仮名", "gold_paths": [
        {"path_id": "canonical", "is_canonical": True, "tokens": tokens}
    ]}


def test_gold_path_is_reachable_when_dictionary_entries_match() -> None:
    dictionary = Dictionary([DictionaryEntry("かな", "仮名", "noun", 1.0)])
    result = enrich_candidate(_candidate([{"reading": "かな", "surface": "仮名"}]), dictionary)
    assert result["gold_reachable"]
    assert result["gold_paths"][0]["reading_valid"]
    assert not result["gold_paths"][0]["repaired"] if "repaired" in result["gold_paths"][0] else True


def test_unreachable_gold_is_kept_and_surface_repair_is_recorded() -> None:
    dictionary = Dictionary([DictionaryEntry("か", "仮", "noun", 1.0), DictionaryEntry("な", "名", "noun", 1.0)])
    result = validate_gold_paths([_candidate([{"reading": "かな", "surface": "仮名"}])], dictionary)[0]
    assert not result["gold_reachable"]
    assert result["gold_paths"][0]["replacement_found"]
    assert result["gold_paths"][0]["tokens"] == [
        {"surface": "仮", "reading": "か", "surface_start": 0, "surface_end": 1,
         "reading_start": 0, "reading_end": 1, "is_literal": False},
        {"surface": "名", "reading": "な", "surface_start": 1, "surface_end": 2,
         "reading_start": 1, "reading_end": 2, "is_literal": False},
    ]


def test_reading_mismatch_is_not_silently_repaired() -> None:
    dictionary = Dictionary([DictionaryEntry("かな", "仮名", "noun", 1.0)])
    result = enrich_candidate(_candidate([{"reading": "か", "surface": "仮名"}]), dictionary)
    assert not result["gold_reading_valid"]
    assert not result["gold_reachable"]
