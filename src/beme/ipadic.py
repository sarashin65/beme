"""Convert MeCab IPADIC source CSV records into BeME dictionary entries."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import tarfile
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, TextIO

from .dictionary import Dictionary, is_hiragana

DEFAULT_RAW_COST_FLOOR = 3_000
DEFAULT_RAW_COST_CEILING = 10_000
DEFAULT_BEME_COST_FLOOR = 1.75
DEFAULT_BEME_COST_CEILING = 5.25

# IPADIC is an analysis dictionary: its katakana surfaces are valid inputs but
# over-produced as IME outputs.  AJIMEE ablation selected this fixed P0 value.
KATAKANA_SURFACE_COST_PENALTY = 1.0


@dataclass(frozen=True, slots=True)
class DictionaryBuildStats:
    source_rows: int
    imported_entries: int
    duplicate_rows: int
    skipped_symbols: int
    skipped_readings: int
    overlay_entries: int
    output_entries: int


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def katakana_to_hiragana(value: str) -> str:
    """Normalize full-/half-width katakana and convert it to hiragana."""

    normalized = unicodedata.normalize("NFKC", value)
    return "".join(
        chr(ord(char) - 0x60) if "\u30a1" <= char <= "\u30f6" else char
        for char in normalized
    )


def _contains_katakana(surface: str) -> bool:
    """Return whether *surface* contains a full- or half-width katakana letter."""

    normalized = unicodedata.normalize("NFKC", surface)
    return any(
        "\u30a1" <= char <= "\u30fa"
        or "\u30fd" <= char <= "\u30ff"
        or "\u31f0" <= char <= "\u31ff"
        for char in normalized
    )


def map_ipadic_pos(primary: str, secondary: str, tertiary: str) -> str:
    """Map IPADIC's Japanese POS hierarchy to BeME's compact POS labels."""

    if primary == "名詞":
        if secondary == "固有名詞":
            return "proper_noun"
        if secondary == "代名詞":
            return "pronoun"
        if secondary == "数":
            return "number"
        if secondary == "接尾" and tertiary == "助数詞":
            return "counter"
        if secondary == "接尾":
            return "suffix"
        return "noun"
    return {
        "動詞": "verb",
        "形容詞": "adjective",
        "助詞": "particle",
        "助動詞": "auxiliary",
        "副詞": "adverb",
        "接続詞": "conjunction",
        "連体詞": "adnominal",
        "感動詞": "interjection",
        "フィラー": "interjection",
        "接頭詞": "prefix",
        "記号": "symbol",
    }.get(primary, "other")


def map_ipadic_cost(
    raw_cost: int,
    *,
    raw_floor: int = DEFAULT_RAW_COST_FLOOR,
    raw_ceiling: int = DEFAULT_RAW_COST_CEILING,
    beme_floor: float = DEFAULT_BEME_COST_FLOOR,
    beme_ceiling: float = DEFAULT_BEME_COST_CEILING,
) -> float:
    """Map an IPADIC word cost to BeME's bounded lower-is-better range.

    Values outside the configured IPADIC range are clamped.  The default BeME
    floor deliberately remains above every hand-maintained overlay cost so the
    established homophone priorities win exact-reading ties.
    """

    if raw_floor >= raw_ceiling:
        raise ValueError("raw cost floor must be lower than raw cost ceiling")
    if beme_floor < 0 or beme_floor >= beme_ceiling:
        raise ValueError("BeME cost floor must be non-negative and lower than its ceiling")
    bounded = min(max(raw_cost, raw_floor), raw_ceiling)
    ratio = (bounded - raw_floor) / (raw_ceiling - raw_floor)
    return round(beme_floor + ratio * (beme_ceiling - beme_floor), 3)


def _rows(handle: TextIO, label: str) -> Iterator[tuple[list[str], str, int]]:
    reader = csv.reader(handle)
    for row in reader:
        yield row, label, reader.line_num


def _archive_rows(source: Path, encoding: str) -> Iterator[tuple[list[str], str, int]]:
    with tarfile.open(source, mode="r:*") as archive:
        members = sorted(
            (member for member in archive.getmembers() if member.isfile() and member.name.endswith(".csv")),
            key=lambda member: member.name,
        )
        if not members:
            raise ValueError(f"{source}: archive contains no IPADIC CSV files")
        for member in members:
            raw = archive.extractfile(member)
            if raw is None:
                raise ValueError(f"{source}:{member.name}: cannot read archive member")
            with _text_reader(raw, encoding) as handle:
                yield from _rows(handle, f"{source}:{member.name}")


def _text_reader(handle: BinaryIO, encoding: str) -> TextIO:
    return io.TextIOWrapper(handle, encoding=encoding, newline="")


def _source_rows(source: Path, encoding: str) -> Iterator[tuple[list[str], str, int]]:
    if source.is_dir():
        csv_paths = sorted(path for path in source.rglob("*.csv") if path.is_file())
        if not csv_paths:
            raise ValueError(f"{source}: directory contains no IPADIC CSV files")
        for path in csv_paths:
            with path.open("r", encoding=encoding, newline="") as handle:
                yield from _rows(handle, str(path))
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    yield from _archive_rows(source, encoding)


def _load_imported_entries(
    source: Path,
    *,
    encoding: str,
    raw_floor: int,
    raw_ceiling: int,
    beme_floor: float,
    beme_ceiling: float,
) -> tuple[dict[tuple[str, str, str], float], tuple[int, int, int, int]]:
    entries: dict[tuple[str, str, str], float] = {}
    source_rows = duplicate_rows = skipped_symbols = skipped_readings = 0
    for row, label, line_number in _source_rows(source, encoding):
        source_rows += 1
        if len(row) < 12:
            raise ValueError(f"{label}: line {line_number}: expected at least 12 CSV fields, got {len(row)}")
        surface = row[0]
        primary, secondary, tertiary = row[4:7]
        if not surface:
            raise ValueError(f"{label}: line {line_number}: surface must be non-empty")
        if primary == "記号":
            skipped_symbols += 1
            continue
        reading = katakana_to_hiragana(row[11])
        if not is_hiragana(reading):
            skipped_readings += 1
            continue
        try:
            raw_cost = int(row[3])
        except ValueError as exc:
            raise ValueError(f"{label}: line {line_number}: invalid IPADIC cost {row[3]!r}") from exc
        pos = map_ipadic_pos(primary, secondary, tertiary)
        cost = map_ipadic_cost(
            raw_cost,
            raw_floor=raw_floor,
            raw_ceiling=raw_ceiling,
            beme_floor=beme_floor,
            beme_ceiling=beme_ceiling,
        )
        if _contains_katakana(surface):
            cost += KATAKANA_SURFACE_COST_PENALTY
        key = (reading, surface, pos)
        previous = entries.get(key)
        if previous is not None:
            duplicate_rows += 1
            if cost >= previous:
                continue
        entries[key] = cost
    return entries, (source_rows, duplicate_rows, skipped_symbols, skipped_readings)


def build_dictionary(
    source: str | Path,
    output: str | Path,
    *,
    overlays: Iterable[str | Path] = (),
    encoding: str = "euc_jp",
    raw_floor: int = DEFAULT_RAW_COST_FLOOR,
    raw_ceiling: int = DEFAULT_RAW_COST_CEILING,
    beme_floor: float = DEFAULT_BEME_COST_FLOOR,
    beme_ceiling: float = DEFAULT_BEME_COST_CEILING,
) -> DictionaryBuildStats:
    """Build a deterministic BeME TSV from IPADIC CSV files and overlays."""

    source_path = Path(source)
    output_path = Path(output)
    entries, counters = _load_imported_entries(
        source_path,
        encoding=encoding,
        raw_floor=raw_floor,
        raw_ceiling=raw_ceiling,
        beme_floor=beme_floor,
        beme_ceiling=beme_ceiling,
    )
    imported_entries = len(entries)
    overlay_entries = 0
    for overlay_path in overlays:
        overlay = Dictionary.from_tsv(overlay_path)
        overlay_entries += len(overlay)
        for entry in overlay.entries:
            entries[(entry.reading, entry.surface, entry.pos)] = entry.cost

    ordered = sorted(
        entries.items(),
        key=lambda item: (item[0][0], item[1], item[0][1], item[0][2]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("reading", "surface", "pos", "cost"))
            for (reading, surface, pos), cost in ordered:
                writer.writerow((reading, surface, pos, f"{cost:.3f}"))
        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    source_rows, duplicate_rows, skipped_symbols, skipped_readings = counters
    return DictionaryBuildStats(
        source_rows=source_rows,
        imported_entries=imported_entries,
        duplicate_rows=duplicate_rows,
        skipped_symbols=skipped_symbols,
        skipped_readings=skipped_readings,
        overlay_entries=overlay_entries,
        output_entries=len(ordered),
    )


__all__ = [
    "DEFAULT_BEME_COST_CEILING",
    "DEFAULT_BEME_COST_FLOOR",
    "DEFAULT_RAW_COST_CEILING",
    "DEFAULT_RAW_COST_FLOOR",
    "DictionaryBuildStats",
    "build_dictionary",
    "katakana_to_hiragana",
    "map_ipadic_cost",
    "map_ipadic_pos",
    "sha256_file",
]
