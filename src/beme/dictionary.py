"""Small, deterministic dictionary and prefix index for BeME."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


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


class _TrieNode:
    __slots__ = ("children", "entries")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.entries: list[DictionaryEntry] = []


class Dictionary:
    """An immutable entry collection backed by a first-character/prefix trie."""

    def __init__(self, entries: Iterable[DictionaryEntry]) -> None:
        ordered = sorted(
            tuple(entries),
            key=lambda entry: (entry.reading, entry.cost, entry.surface, entry.pos),
        )
        self.entries: tuple[DictionaryEntry, ...] = ordered
        self.first_char_index: Mapping[str, tuple[DictionaryEntry, ...]] = {
            first: tuple(
                sorted(
                    values,
                    key=lambda entry: (len(entry.reading), entry.cost, entry.surface, entry.pos),
                )
            )
            for first, values in self._group_by_first(ordered).items()
        }

        root = _TrieNode()
        for entry in ordered:
            node = root
            for char in entry.reading:
                node = node.children.setdefault(char, _TrieNode())
            node.entries.append(entry)
        self._root = root

        # This public view is useful for diagnostics and keeps prefix lookup
        # explicit, while matches_at traverses the trie without making copies.
        prefix_values: dict[str, list[DictionaryEntry]] = {}
        for entry in ordered:
            for end in range(1, len(entry.reading) + 1):
                prefix_values.setdefault(entry.reading[:end], []).append(entry)
        self.prefix_index: Mapping[str, tuple[DictionaryEntry, ...]] = {
            prefix: tuple(
                sorted(values, key=lambda entry: (entry.reading, entry.cost, entry.surface, entry.pos))
            )
            for prefix, values in prefix_values.items()
        }
        self.by_first_char = self.first_char_index
        self.by_prefix = self.prefix_index
        self._entries_by_reading: dict[str, tuple[DictionaryEntry, ...]] = {}
        for entry in ordered:
            self._entries_by_reading.setdefault(entry.reading, tuple())
            self._entries_by_reading[entry.reading] = self._entries_by_reading[entry.reading] + (entry,)

    @staticmethod
    def _group_by_first(entries: Iterable[DictionaryEntry]) -> dict[str, list[DictionaryEntry]]:
        groups: dict[str, list[DictionaryEntry]] = {}
        for entry in entries:
            groups.setdefault(entry.reading[0], []).append(entry)
        return groups

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
                entries.append(DictionaryEntry(reading, surface, pos, cost))

        if not entries:
            raise ValueError(f"{source}: no dictionary entries found")
        return cls(entries)

    def entries_for_reading(self, reading: str) -> tuple[DictionaryEntry, ...]:
        """Return exact-reading entries in deterministic order."""

        return self._entries_by_reading.get(reading, tuple())

    lookup = entries_for_reading

    def entries_with_prefix(self, prefix: str) -> tuple[DictionaryEntry, ...]:
        """Return entries whose readings start with *prefix*."""

        return self.prefix_index.get(prefix, tuple())

    def matches_at(self, reading: str, position: int) -> tuple[DictionaryEntry, ...]:
        """Return dictionary entries beginning at an input position.

        The trie stops as soon as the input ceases to match, avoiding a scan of
        unrelated readings.
        """

        if position < 0 or position >= len(reading):
            return tuple()
        node = self._root
        matches: list[DictionaryEntry] = []
        for char in reading[position:]:
            node = node.children.get(char)
            if node is None:
                break
            if node.entries:
                matches.extend(node.entries)
        return tuple(
            sorted(matches, key=lambda entry: (len(entry.reading), entry.cost, entry.surface, entry.pos))
        )

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
