from __future__ import annotations

import csv
import io
import tarfile
from pathlib import Path

import pytest

from beme.dictionary import Dictionary
from beme.ipadic import (
    build_dictionary,
    katakana_to_hiragana,
    map_ipadic_cost,
    map_ipadic_pos,
    sha256_file,
)


def _ipadic_row(
    surface: str,
    cost: int,
    primary: str,
    reading: str,
    secondary: str = "一般",
    tertiary: str = "*",
) -> list[str]:
    return [
        surface,
        "1",
        "1",
        str(cost),
        primary,
        secondary,
        tertiary,
        "*",
        "*",
        "*",
        surface,
        reading,
        reading,
    ]


def _write_source_archive(path: Path, rows: list[list[str]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    payload = buffer.getvalue().encode("utf-8")
    info = tarfile.TarInfo("ipadic/Noun.csv")
    info.size = len(payload)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))


def test_katakana_cost_and_pos_mapping() -> None:
    assert katakana_to_hiragana("ｷｼｬー") == "きしゃー"
    assert map_ipadic_cost(3_000) == 1.75
    assert map_ipadic_cost(6_500) == 3.5
    assert map_ipadic_cost(10_000) == 5.25
    assert map_ipadic_cost(-10_000) == 1.75
    assert map_ipadic_cost(30_000) == 5.25
    assert map_ipadic_pos("名詞", "固有名詞", "人名") == "proper_noun"
    assert map_ipadic_pos("名詞", "接尾", "助数詞") == "counter"
    assert map_ipadic_pos("動詞", "自立", "*") == "verb"
    with pytest.raises(ValueError, match="raw cost floor"):
        map_ipadic_cost(5_000, raw_floor=10, raw_ceiling=10)


def test_build_dictionary_merges_overlay_and_filters_invalid_rows(tmp_path: Path) -> None:
    source = tmp_path / "ipadic.tar.gz"
    _write_source_archive(
        source,
        [
            _ipadic_row("記者", 5_000, "名詞", "キシャ"),
            _ipadic_row("記者", 4_000, "名詞", "キシャ"),
            _ipadic_row("汽車", 9_000, "名詞", "キシャ"),
            _ipadic_row("。", 1_000, "記号", "。"),
            _ipadic_row("invalid", 1_000, "名詞", "ABC"),
        ],
    )
    overlay = tmp_path / "overlay.tsv"
    overlay.write_text(
        "reading\tsurface\tpos\tcost\n"
        "きしゃ\t記者\tnoun\t0.800\n"
        "こうしょう\t考証\tnoun\t1.150\n",
        encoding="utf-8",
    )
    output = tmp_path / "dictionary.tsv"

    stats = build_dictionary(source, output, overlays=[overlay], encoding="utf-8")

    assert stats.source_rows == 5
    assert stats.imported_entries == 2
    assert stats.duplicate_rows == 1
    assert stats.skipped_symbols == 1
    assert stats.skipped_readings == 1
    assert stats.overlay_entries == 2
    assert stats.output_entries == 3
    dictionary = Dictionary.from_tsv(output)
    assert [(entry.surface, entry.cost) for entry in dictionary.lookup("きしゃ")] == [
        ("記者", 0.8),
        ("汽車", 4.75),
    ]
    assert dictionary.lookup("こうしょう")[0].surface == "考証"
    assert len(sha256_file(source)) == 64


def test_build_dictionary_rejects_malformed_source_row(tmp_path: Path) -> None:
    source = tmp_path / "ipadic.tar.gz"
    _write_source_archive(source, [["too", "short"]])

    with pytest.raises(ValueError, match="expected at least 12 CSV fields"):
        build_dictionary(source, tmp_path / "dictionary.tsv", encoding="utf-8")
