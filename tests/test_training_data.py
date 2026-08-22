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
