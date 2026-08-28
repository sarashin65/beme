"""Deterministic (reading, surface) corpus frequency table for F2."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"BEMECF1"
VERSION = 1
DEFAULT_CORPUS_FREQ_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "corpus_freq_v1.bin"
)
_HEADER = struct.Struct("<8sHII")


class CorpusFrequencyError(ValueError):
    """Raised when the table file is missing, truncated, or malformed."""


@dataclass(frozen=True, slots=True)
class CorpusFrequencyTable:
    """Counts of (reading, surface) pairs over train-split Wikipedia articles.

    Readings are hiragana-only (``build_d1_data.node_reading`` semantics);
    surfaces are the raw node surfaces. All counts are token counts.
    """

    readings: tuple[str, ...]
    surfaces: tuple[str, ...]
    pair_counts: tuple[int, ...]  # aligned with pair_readings / pair_surfaces
    pair_readings: tuple[int, ...]  # index into readings
    pair_surfaces: tuple[int, ...]  # index into surfaces
    surface_counts: tuple[int, ...]  # aligned with surfaces
    total_tokens: int
    _reading_index: dict[str, int] = field(init=False, repr=False)
    _surface_index: dict[str, int] = field(init=False, repr=False)
    _pair_index: dict[tuple[int, int], int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_reading_index", {r: i for i, r in enumerate(self.readings)}
        )
        object.__setattr__(
            self, "_surface_index", {s: i for i, s in enumerate(self.surfaces)}
        )
        object.__setattr__(
            self,
            "_pair_index",
            {
                (ri, si): i
                for i, (ri, si) in enumerate(zip(self.pair_readings, self.pair_surfaces))
            },
        )

    # -- lookups used by task4's feature implementation -------------------
    def pair_count(self, reading: str, surface: str) -> int:
        """c(r, s): tokens where *surface* was read *reading*."""
        ri = self._reading_index.get(reading)
        si = self._surface_index.get(surface)
        if ri is None or si is None:
            return 0
        index = self._pair_index.get((ri, si))
        return self.pair_counts[index] if index is not None else 0

    def reading_count(self, reading: str) -> int:
        """c(r) = sum_s c(r, s)."""
        ri = self._reading_index.get(reading)
        if ri is None:
            return 0
        return sum(
            count
            for index, count in enumerate(self.pair_counts)
            if self.pair_readings[index] == ri
        )

    def surface_count(self, surface: str) -> int:
        """c(s): tokens with this surface, any reading."""
        si = self._surface_index.get(surface)
        return self.surface_counts[si] if si is not None else 0

    # -- construction ------------------------------------------------------
    @classmethod
    def from_counts(
        cls,
        pairs: dict[tuple[str, str], int],
        surfaces: dict[str, int],
        total: int,
    ) -> "CorpusFrequencyTable":
        """Build an in-memory table from raw counters (test / builder input).

        ``pairs`` maps (reading, surface) -> count; ``surfaces`` maps
        surface -> count; ``total`` is the total token count N.
        """
        readings = sorted({reading for reading, _ in pairs})
        surface_names = sorted(surfaces)
        reading_index = {r: i for i, r in enumerate(readings)}
        surface_index = {s: i for i, s in enumerate(surface_names)}
        ordered = sorted(
            ((reading, surface, count) for (reading, surface), count in pairs.items() if count > 0),
            key=lambda item: (reading_index[item[0]], surface_index[item[1]]),
        )
        pair_counts = tuple(count for _, _, count in ordered)
        pair_readings = tuple(reading_index[reading] for reading, _, _ in ordered)
        pair_surfaces = tuple(surface_index[surface] for _, surface, _ in ordered)
        surface_counts = tuple(surfaces[s] for s in surface_names)
        return cls(
            readings=tuple(readings),
            surfaces=tuple(surface_names),
            pair_counts=pair_counts,
            pair_readings=pair_readings,
            pair_surfaces=pair_surfaces,
            surface_counts=surface_counts,
            total_tokens=total,
        )

    # -- binary format -----------------------------------------------------
    def dump(self, path: str | Path) -> str:
        """Write the table deterministically; returns the SHA-256 hex digest.

        Layout: header (magic, version, n_readings, n_surfaces, n_pairs),
        then per-section length-prefixed UTF-8 string arrays and little-endian
        uint32 count arrays. Identical inputs produce identical bytes.
        """
        path = Path(path)
        buffer = bytearray()
        buffer += _HEADER.pack(MAGIC, VERSION, len(self.readings), len(self.surfaces))
        buffer += _u32(len(self.pair_counts))
        buffer += _strings(self.readings)
        buffer += _strings(self.surfaces)
        buffer += _u32s(self.pair_readings)
        buffer += _u32s(self.pair_surfaces)
        buffer += _u32s(self.pair_counts)
        buffer += _u32s(self.surface_counts)
        buffer += _u64(self.total_tokens)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(buffer))
        return hashlib.sha256(bytes(buffer)).hexdigest()

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CORPUS_FREQ_PATH) -> "CorpusFrequencyTable":
        """Load a table; raises CorpusFrequencyError if the file is absent."""
        path = Path(path)
        if not path.exists():
            raise CorpusFrequencyError(f"corpus frequency table not found: {path}")
        data = path.read_bytes()
        offset = 0

        def take(size: int) -> bytes:
            nonlocal offset
            chunk = data[offset : offset + size]
            if len(chunk) != size:
                raise CorpusFrequencyError("corpus frequency table is truncated")
            offset += size
            return chunk

        magic, version, n_readings, n_surfaces = _HEADER.unpack(take(_HEADER.size))
        if magic.rstrip(b"\x00") != MAGIC or version != VERSION:
            raise CorpusFrequencyError("unsupported corpus frequency table format")
        n_pairs = struct.unpack("<I", take(4))[0]
        readings = _read_strings(take, n_readings)
        surfaces = _read_strings(take, n_surfaces)
        pair_readings = _read_u32s(take, n_pairs)
        pair_surfaces = _read_u32s(take, n_pairs)
        pair_counts = _read_u32s(take, n_pairs)
        surface_counts = _read_u32s(take, n_surfaces)
        total_tokens = struct.unpack("<Q", take(8))[0]
        if offset != len(data):
            raise CorpusFrequencyError("corpus frequency table has trailing data")
        table = cls(
            readings=tuple(readings),
            surfaces=tuple(surfaces),
            pair_counts=tuple(pair_counts),
            pair_readings=tuple(pair_readings),
            pair_surfaces=tuple(pair_surfaces),
            surface_counts=tuple(surface_counts),
            total_tokens=total_tokens,
        )
        if table.total_tokens != sum(table.pair_counts):
            raise CorpusFrequencyError("total_tokens does not match sum of pair counts")
        return table


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _strings(values: tuple[str, ...]) -> bytes:
    buffer = bytearray()
    for value in values:
        encoded = value.encode("utf-8")
        buffer += _u32(len(encoded))
        buffer += encoded
    return bytes(buffer)


def _u32s(values: tuple[int, ...]) -> bytes:
    return struct.pack(f"<{len(values)}I", *values) if values else b""


def _read_strings(take, count: int) -> list[str]:
    values = []
    for _ in range(count):
        length = struct.unpack("<I", take(4))[0]
        values.append(take(length).decode("utf-8"))
    return values


def _read_u32s(take, count: int) -> list[int]:
    if count == 0:
        return []
    return list(struct.unpack(f"<{count}I", take(4 * count)))


__all__ = [
    "DEFAULT_CORPUS_FREQ_PATH",
    "MAGIC",
    "VERSION",
    "CorpusFrequencyError",
    "CorpusFrequencyTable",
]
