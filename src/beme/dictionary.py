"""Deterministic BeME dictionary and memory-conscious exact-reading index."""

from __future__ import annotations

import csv
import math
import sys
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_REQUIRED_COLUMNS = ("reading", "surface", "pos", "cost")
_OPTIONAL_COLUMNS = (
    "pos_detail",
    "conjugation",
    "left_id",
    "right_id",
    "source",
    "raw_cost",
)
_OPTIONAL_COLUMN_SET = frozenset(_OPTIONAL_COLUMNS)
_SOURCES = frozenset(("ipadic", "overlay", "learned", "manual"))


def is_hiragana(text: str) -> bool:
    """Return whether *text* consists only of hiragana and ``ー``."""

    return bool(text) and all("\u3041" <= char <= "\u3096" or char == "ー" for char in text)


def _parse_non_negative_int(
    value: str | None,
    name: str,
    source: Path,
    line_number: int,
    *,
    default: int,
) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{source}: line {line_number}: {name} must be a non-negative integer, got {value!r}"
        ) from exc
    if parsed < 0:
        raise ValueError(
            f"{source}: line {line_number}: {name} must be a non-negative integer, got {value!r}"
        )
    return parsed


def _parse_optional_int(
    value: str | None,
    name: str,
    source: Path,
    line_number: int,
) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{source}: line {line_number}: {name} must be an integer, got {value!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class DictionaryEntry:
    """One lexical item in the Phase 1 dictionary."""

    reading: str
    surface: str
    pos: str
    cost: float
    pos_detail: str = ""
    conjugation: str = ""
    left_id: int = -1
    right_id: int = -1
    source: str = ""
    raw_cost: int | None = None



class Dictionary:
    """An immutable entry collection backed by an exact-reading index."""

    def __init__(self, entries: Iterable[DictionaryEntry]) -> None:
        ordered = tuple(
            sorted(
                entries,
                key=lambda entry: (entry.reading, entry.cost, entry.surface, entry.pos),
            )
        )
        self.entries: tuple[DictionaryEntry, ...] = ordered
        grouped: dict[str, list[DictionaryEntry]] = {}
        for entry in ordered:
            grouped.setdefault(entry.reading, []).append(entry)
        self._entries_by_reading: dict[str, tuple[DictionaryEntry, ...]] = {
            reading: tuple(values) for reading, values in grouped.items()
        }
        self._readings = tuple(self._entries_by_reading)
        self.reading_conditional: dict[str, tuple[float, int]] = {}
        from .ipadic import READING_CONDITIONAL_TEMPERATURE

        for reading, reading_entries in self._entries_by_reading.items():
            raw_costs = [entry.raw_cost for entry in reading_entries if entry.raw_cost is not None]
            if raw_costs:
                scaled = [-raw / READING_CONDITIONAL_TEMPERATURE for raw in raw_costs]
                maximum = max(scaled)
                log_partition = maximum + math.log(
                    sum(math.exp(value - maximum) for value in scaled)
                )
            else:
                log_partition = 0.0
            self.reading_conditional[reading] = (log_partition, len(reading_entries))

    @classmethod
    def from_tsv(cls, path: str | Path) -> "Dictionary":
        """Load a dictionary TSV with four required and optional columns.

        Errors name both the source path and physical line, so malformed data
        is straightforward to fix rather than silently producing an empty index.
        Blank lines are ignored.  The first four header columns are required in
        fixed order; known optional columns may follow in any order.
        """

        source = Path(path)
        entries: list[DictionaryEntry] = []
        try:
            handle = source.open("r", encoding="utf-8", newline="")
        except OSError:
            raise
        with handle:
            reader = csv.reader(handle, delimiter="\t", strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(
                    f"{source}: line 1: missing TSV header; expected reading\\tsurface\\tpos\\tcost"
                ) from exc
            except csv.Error as exc:
                raise ValueError(f"{source}: line 1: malformed TSV header: {exc}") from exc
            if header:
                header[0] = header[0].removeprefix("\ufeff")
            if tuple(header[: len(_REQUIRED_COLUMNS)]) != _REQUIRED_COLUMNS:
                expected = "\\t".join(_REQUIRED_COLUMNS)
                actual = "\\t".join(header)
                raise ValueError(f"{source}: line 1: expected header starting with {expected!r}, got {actual!r}")
            optional_header = header[len(_REQUIRED_COLUMNS) :]
            if len(set(header)) != len(header):
                raise ValueError(f"{source}: line 1: duplicate TSV column name")
            unknown = [name for name in optional_header if name not in _OPTIONAL_COLUMN_SET]
            if unknown:
                raise ValueError(f"{source}: line 1: unknown TSV column {unknown[0]!r}")

            while True:
                try:
                    row = next(reader)
                except StopIteration:
                    break
                except csv.Error as exc:
                    raise ValueError(f"{source}: line {reader.line_num}: malformed TSV: {exc}") from exc
                line_number = reader.line_num
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError(
                        f"{source}: line {line_number}: expected {len(header)} tab-separated fields, got {len(row)}"
                    )
                fields = dict(zip(header, row, strict=True))
                reading, surface, pos, raw_cost = (
                    fields["reading"],
                    fields["surface"],
                    fields["pos"],
                    fields["cost"],
                )
                if not is_hiragana(reading):
                    raise ValueError(
                        f"{source}: line {line_number}: reading must be non-empty hiragana (ー allowed): {reading!r}"
                    )
                if not surface:
                    raise ValueError(f"{source}: line {line_number}: surface must be non-empty")
                if not pos:
                    raise ValueError(f"{source}: line {line_number}: pos must be non-empty")
                try:
                    cost = float(raw_cost)
                except ValueError as exc:
                    raise ValueError(
                        f"{source}: line {line_number}: cost must be a finite number, got {raw_cost!r}"
                    ) from exc
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError(
                        f"{source}: line {line_number}: cost must be a finite non-negative number, got {raw_cost!r}"
                    )
                pos_detail = fields.get("pos_detail", "")
                conjugation = fields.get("conjugation", "")
                left_id = _parse_non_negative_int(fields.get("left_id"), "left_id", source, line_number, default=-1)
                right_id = _parse_non_negative_int(fields.get("right_id"), "right_id", source, line_number, default=-1)
                entry_source = fields.get("source", "")
                if entry_source and entry_source not in _SOURCES:
                    raise ValueError(
                        f"{source}: line {line_number}: source must be one of {sorted(_SOURCES)!r}, got {entry_source!r}"
                    )
                raw_cost_value = _parse_optional_int(fields.get("raw_cost"), "raw_cost", source, line_number)
                entries.append(
                    DictionaryEntry(
                        sys.intern(reading),
                        sys.intern(surface),
                        sys.intern(pos),
                        cost,
                        sys.intern(pos_detail),
                        sys.intern(conjugation),
                        left_id,
                        right_id,
                        sys.intern(entry_source),
                        raw_cost_value,
                    )
                )

        if not entries:
            raise ValueError(f"{source}: no dictionary entries found")
        return cls(entries)

    def entries_for_reading(self, reading: str) -> tuple[DictionaryEntry, ...]:
        """Return exact-reading entries in deterministic order."""

        return self._entries_by_reading.get(reading, tuple())

    lookup = entries_for_reading

    def entries_with_prefix(self, prefix: str) -> tuple[DictionaryEntry, ...]:
        """Return entries whose readings start with *prefix*."""

        if not prefix:
            return tuple()
        index = bisect_left(self._readings, prefix)
        matches: list[DictionaryEntry] = []
        while index < len(self._readings):
            reading = self._readings[index]
            if not reading.startswith(prefix):
                break
            matches.extend(self._entries_by_reading[reading])
            index += 1
        return tuple(matches)

    def matches_at(self, reading: str, position: int) -> tuple[DictionaryEntry, ...]:
        """Return dictionary entries beginning at an input position.

        Input strings are short, so probing their possible end positions against
        the exact-reading index is both faster and substantially smaller than a
        per-character Python trie for a 350k-entry IPADIC dictionary.
        """

        if position < 0 or position >= len(reading):
            return tuple()
        matches: list[DictionaryEntry] = []
        for end in range(position + 1, len(reading) + 1):
            matches.extend(self._entries_by_reading.get(reading[position:end], ()))
        return tuple(matches)

    def __len__(self) -> int:
        return len(self.entries)


MiniDictionary = Dictionary
load_tsv = Dictionary.from_tsv

__all__ = [
    "Dictionary",
    "DictionaryEntry",
    "MiniDictionary",
    "is_hiragana",
    "load_tsv",
]
