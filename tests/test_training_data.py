"""Contract tests for strict training JSONL and split manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from beme.training_data import TrainingExample, load_jsonl, load_splits


FIXTURE = Path(__file__).parent / "data" / "phase2"


def write_jsonl(tmp_path: Path, text: str, name: str = "data.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_canonical_and_positive_alias_create_same_example() -> None:
    assert load_jsonl(FIXTURE / "train.jsonl")[0].target_surface == "仮名"
    assert load_jsonl(FIXTURE / "dev.jsonl")[0] == TrainingExample("ねこ", "猫", id="dev-1", source="synthetic", article_id="article-dev")


def test_both_target_keys_must_match(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, '{"reading":"か","target_surface":"仮名","positive":"佳名"}\n')
    with pytest.raises(ValueError, match=r"line 1.*must be equal"):
        load_jsonl(path)


@pytest.mark.parametrize("text, line", [
    ("\n", 1),
    ('{"reading":"か","target_surface":"可"}\n\n', 2),
    ('{"reading":"か","target_surface":"可","reading":"課"}\n', 1),
    ('{"reading":"か","target_surface":"可","extra":1}\n', 1),
    ('{"reading":1,"target_surface":"可"}\n', 1),
])
def test_malformed_rows_include_line_number(tmp_path: Path, text: str, line: int) -> None:
    with pytest.raises(ValueError, match=rf"line {line}"):
        load_jsonl(write_jsonl(tmp_path, text))


def test_duplicate_id_and_invalid_utf8_include_line_number(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, '{"id":"x","reading":"か","target_surface":"可"}\n{"id":"x","reading":"き","target_surface":"気"}\n')
    with pytest.raises(ValueError, match=r"line 2.*duplicate id"):
        load_jsonl(path)
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_bytes(b'{"reading":"\xff","target_surface":"x"}\n')
    with pytest.raises(ValueError, match=r"line 1.*UTF-8"):
        load_jsonl(invalid)


def test_split_leakage_and_missing_article_ids_are_not_silently_accepted(tmp_path: Path) -> None:
    def split(name: str, article: str | None) -> Path:
        field = f',"article_id":"{article}"' if article else ""
        return write_jsonl(tmp_path, f'{{"reading":"か","target_surface":"可"{field}}}\n', name)

    paths = {"train": split("train.jsonl", "same"), "dev": split("dev.jsonl", "same"), "test": split("test.jsonl", "other")}
    with pytest.raises(ValueError, match="crosses"):
        load_splits(paths)
    paths["dev"] = split("dev.jsonl", None)
    splits, manifest = load_splits(paths)
    assert len(splits["train"]) == 1
    assert manifest["article_disjoint_verified"] is False
    assert manifest["splits"]["train"]["article_disjoint_verified"] is False


def test_manifest_is_byte_stable_and_records_fixed_fixture_sha256() -> None:
    paths = {name: FIXTURE / f"{name}.jsonl" for name in ("train", "dev", "test")}
    _, first = load_splits(paths)
    _, second = load_splits(paths)
    assert first == second
    assert first["splits"]["train"]["examples"] == 1
    assert first["splits"]["train"]["sha256"] == hashlib.sha256(paths["train"].read_bytes()).hexdigest()
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(second, ensure_ascii=False, sort_keys=True)


def test_missing_split_path_fails_instead_of_using_synthetic_data(tmp_path: Path) -> None:
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("train", "dev", "test")}
    paths["train"].write_text('{"reading":"か","target_surface":"可"}\n', encoding="utf-8")
    with pytest.raises(OSError):
        load_splits(paths)


def test_all_d1_fields_load_and_positive_is_canonical(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path,
        '{"id":"full-1","source":"synthetic","article_id":"article-full",'
        '"article_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"left_context":"前","reading":"かな","right_context":"後",'
        '"positive":"仮名","candidates":["仮名","佳名"],"viterbi_costs":[1.0,2.0],'
        '"frequency_bucket":{"positive":"mid"},"context_truncated":false,'
        '"contrast_pair_id":"pair-1","gold_index":0,"span_tokens":1,"split":"train"}\n',
    )
    example = load_jsonl(path)[0]
    assert example.target_surface == "仮名"
    assert example.article_sha256 == "a" * 64
    assert example.context_truncated is False
    assert example.contrast_pair_id == "pair-1"
    assert example.gold_index == 0
    assert example.span_tokens == 1
    assert example.split == "train"


def test_unknown_field_is_still_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"line 1.*unknown field"):
        load_jsonl(write_jsonl(tmp_path, '{"reading":"か","positive":"可","new_field":1}\n'))


@pytest.mark.parametrize("value", ["a", "g" * 64, "1" * 63])
def test_article_sha256_format_is_strict(tmp_path: Path, value: str) -> None:
    text = json.dumps({"reading": "か", "positive": "可", "article_sha256": value}) + "\n"
    with pytest.raises(ValueError, match=r"line 1.*article_sha256"):
        load_jsonl(write_jsonl(tmp_path, text))


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_context_truncated_requires_a_real_bool(tmp_path: Path, value: object) -> None:
    text = json.dumps({"reading": "か", "positive": "可", "context_truncated": value}) + "\n"
    if isinstance(value, bool):
        assert load_jsonl(write_jsonl(tmp_path, text))[0].context_truncated is value
    else:
        with pytest.raises(ValueError, match=r"line 1.*context_truncated"):
            load_jsonl(write_jsonl(tmp_path, text))


def test_contrast_pair_cannot_cross_splits(tmp_path: Path) -> None:
    def make(name: str, split: str) -> Path:
        return write_jsonl(tmp_path, json.dumps({"reading": "か", "positive": "可", "contrast_pair_id": "p", "split": split}) + "\n", name)

    with pytest.raises(ValueError, match="contrast_pair_id.*crosses"):
        load_splits({"train": make("train", "train"), "dev": make("dev", "dev"), "test": make("test", "test")})


def test_same_article_cannot_have_different_hashes(tmp_path: Path) -> None:
    def make(name: str, digest: str) -> Path:
        text = json.dumps({"reading": "か", "positive": "可", "article_id": "article", "article_sha256": digest}) + "\n"
        return write_jsonl(tmp_path, text, name)

    paths = {"train": make("train", "a" * 64), "dev": make("dev", "b" * 64), "test": make("test", "a" * 64)}
    with pytest.raises(ValueError, match="inconsistent article_sha256"):
        load_splits(paths)


@pytest.mark.parametrize("gold_index", [-1, 2])
def test_gold_index_range_is_checked(tmp_path: Path, gold_index: int) -> None:
    text = json.dumps({"reading": "か", "positive": "可", "candidates": ["可", "課"], "viterbi_costs": [1, 2], "gold_index": gold_index}) + "\n"
    with pytest.raises(ValueError, match=r"line 1.*gold_index"):
        load_jsonl(write_jsonl(tmp_path, text))


def test_gold_index_must_point_at_positive(tmp_path: Path) -> None:
    text = '{"reading":"か","positive":"可","candidates":["課","可"],"viterbi_costs":[1,2],"gold_index":0}\n'
    with pytest.raises(ValueError, match=r"line 1.*gold_index"):
        load_jsonl(write_jsonl(tmp_path, text))


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1", []])
def test_span_tokens_must_be_positive_int(tmp_path: Path, value: object) -> None:
    text = json.dumps({"reading": "か", "positive": "可", "span_tokens": value}) + "\n"
    with pytest.raises(ValueError, match=r"line 1.*span_tokens"):
        load_jsonl(write_jsonl(tmp_path, text))


def test_split_value_and_destination_must_match(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, '{"reading":"か","positive":"可","split":"dev"}\n')
    valid = write_jsonl(tmp_path, '{"reading":"き","positive":"気"}\n', "valid.jsonl")
    paths = {"train": path, "dev": valid, "test": valid}
    with pytest.raises(ValueError, match="does not match"):
        load_splits(paths)


def test_split_value_must_be_one_of_the_three_splits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"line 1.*split"):
        load_jsonl(write_jsonl(tmp_path, '{"reading":"か","positive":"可","split":"holdout"}\n'))


def test_manifest_new_fields_are_deterministic(tmp_path: Path) -> None:
    paths = {name: FIXTURE / f"{name}.jsonl" for name in ("train", "dev", "test")}
    first = load_splits(paths)[1]
    second = load_splits(paths)[1]
    assert first == second
    assert first["split_field_verified"] is False
    assert first["splits"]["train"]["article_sha256_present_count"] == 0
    assert first["splits"]["train"]["span_tokens_min"] is None
    assert first["splits"]["train"]["span_tokens_max"] is None


def test_manifest_records_span_token_min_and_max(tmp_path: Path) -> None:
    paths = {}
    for name, value in (("train", 1), ("dev", 3), ("test", 2)):
        paths[name] = write_jsonl(
            tmp_path,
            json.dumps({"reading": "か", "positive": "可", "span_tokens": value}) + "\n",
            f"{name}.jsonl",
        )
    manifest = load_splits(paths)[1]
    assert manifest["splits"]["train"]["span_tokens_min"] == 1
    assert manifest["splits"]["train"]["span_tokens_max"] == 1
