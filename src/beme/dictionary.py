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


def is_hiragana(text: str) -> bool:
    """Return whether *text* consists only of hiragana and ``ー``."""

    return bool(text) and all("\u3041" <= char <= "\u3096" or char == "ー" for char in text)


@dataclass(frozen=True, slots=True)
class DictionaryEntry:
    """One lexical item in the Phase 1 dictionary."""

    reading: str
    surface: str
    pos: str
    cost: float



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

    @classmethod
    def from_tsv(cls, path: str | Path) -> "Dictionary":
        """Load a strict ``reading<TAB>surface<TAB>pos<TAB>cost`` TSV file.

        Errors name both the source path and physical line, so malformed data
        is straightforward to fix rather than silently producing an empty index.
        Blank lines are ignored; all non-blank records must have exactly four
        fields and the header must be the required four columns in that order.
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
            if tuple(header) != _REQUIRED_COLUMNS:
                expected = "\\t".join(_REQUIRED_COLUMNS)
                actual = "\\t".join(header)
                raise ValueError(f"{source}: line 1: expected header {expected!r}, got {actual!r}")

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
                if len(row) != len(_REQUIRED_COLUMNS):
                    raise ValueError(
                        f"{source}: line {line_number}: expected 4 tab-separated fields, got {len(row)}"
                    )
                reading, surface, pos, raw_cost = row
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
                entries.append(
                    DictionaryEntry(
                        sys.intern(reading),
                        sys.intern(surface),
                        sys.intern(pos),
                        cost,
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
