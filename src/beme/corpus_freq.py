"""Deterministic corpus frequency table used by the F2 features."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import struct
from typing import Mapping

MAGIC = b"BEMECF1"
VERSION = 1
DEFAULT_CORPUS_FREQ_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus_freq_v1.bin"
_HEADER = struct.Struct("<7sBHHQIII")


@dataclass(frozen=True, slots=True)
class CorpusFrequencyTable:
    total_tokens: int
    _pairs: tuple[tuple[str, str, int], ...]
    _surfaces: tuple[tuple[str, int], ...]
    _readings: tuple[tuple[str, int], ...]
    _pair_indices: dict[tuple[str, str], int] = field(init=False, repr=False)
    _surface_indices: dict[str, int] = field(init=False, repr=False)
    _reading_indices: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_pair_indices", {(r, s): c for r, s, c in self._pairs})
        object.__setattr__(self, "_surface_indices", dict(self._surfaces))
        object.__setattr__(self, "_reading_indices", dict(self._readings))

    def pair_count(self, reading: str, surface: str) -> int:
        return self._pair_indices.get((reading, surface), 0)

    def reading_count(self, reading: str) -> int:
        return self._reading_indices.get(reading, 0)

    def surface_count(self, surface: str) -> int:
        return self._surface_indices.get(surface, 0)

    @classmethod
    def from_counts(
        cls,
        pairs: Mapping[tuple[str, str], int],
        surfaces: Mapping[str, int],
        total: int,
    ) -> "CorpusFrequencyTable":
        if total < 0:
            raise ValueError("total_tokens must be non-negative")
        pair_rows = []
        reading_counts: dict[str, int] = {}
        for (reading, surface), count in pairs.items():
            if not isinstance(reading, str) or not isinstance(surface, str) or count <= 0:
                raise ValueError("pair keys must be strings and counts must be positive")
            pair_rows.append((reading, surface, int(count)))
            reading_counts[reading] = reading_counts.get(reading, 0) + int(count)
        surface_rows = []
        for surface, count in surfaces.items():
            if not isinstance(surface, str) or count < 0:
                raise ValueError("surface counts must be non-negative")
            surface_rows.append((surface, int(count)))
        return cls(
            int(total),
            tuple(sorted(pair_rows, key=lambda row: (row[0], row[1]))),
            tuple(sorted(surface_rows)),
            tuple(sorted(reading_counts.items())),
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CORPUS_FREQ_PATH) -> "CorpusFrequencyTable":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"corpus frequency table not found: {path}; regenerate with scripts/build_corpus_freq.py"
            )
        try:
            data = path.read_bytes()
            offset = 0

            def take(size: int) -> bytes:
                nonlocal offset
                chunk = data[offset : offset + size]
                if len(chunk) != size:
                    raise ValueError("corpus frequency table is truncated")
                offset += size
                return chunk

            magic, version, reserved, reserved2, total, n_readings, n_surfaces, n_pairs = _HEADER.unpack(
                take(_HEADER.size)
            )
            if magic != MAGIC or version != VERSION or reserved != 0 or reserved2 != 0:
                raise ValueError("unsupported corpus frequency table format")

            def strings(count: int) -> tuple[str, ...]:
                lengths = struct.unpack(f"<{count}I", take(4 * count)) if count else ()
                blob = take(sum(lengths))
                values = []
                cursor = 0
                for length in lengths:
                    try:
                        values.append(blob[cursor : cursor + length].decode("utf-8"))
                    except UnicodeDecodeError as error:
                        raise ValueError("corpus frequency string blob is not UTF-8") from error
                    cursor += length
                if tuple(sorted(set(values))) != tuple(values):
                    raise ValueError("corpus frequency strings must be sorted and unique")
                return tuple(values)

            readings = strings(n_readings)
            surfaces = strings(n_surfaces)
            reading_counts = struct.unpack(f"<{n_readings}Q", take(8 * n_readings)) if n_readings else ()
            surface_counts = struct.unpack(f"<{n_surfaces}Q", take(8 * n_surfaces)) if n_surfaces else ()
            pair_values = struct.unpack(f"<{n_pairs * 3}I", take(12 * n_pairs)) if n_pairs else ()
            if offset != len(data):
                raise ValueError("corpus frequency table has trailing data")
            pairs = []
            previous = None
            for index in range(n_pairs):
                reading_index, surface_index, count = pair_values[index * 3 : index * 3 + 3]
                if reading_index >= n_readings or surface_index >= n_surfaces or count == 0:
                    raise ValueError("invalid corpus frequency pair")
                key = (reading_index, surface_index)
                if previous is not None and key <= previous:
                    raise ValueError("corpus frequency pairs must be sorted and unique")
                previous = key
                pairs.append((readings[reading_index], surfaces[surface_index], count))
            return cls(
                total,
                tuple(pairs),
                tuple(zip(surfaces, surface_counts)),
                tuple(zip(readings, reading_counts)),
            )
        except struct.error as error:
            raise ValueError("invalid corpus frequency table") from error

    def dump(self, path: str | Path) -> str:
        readings = tuple(reading for reading, _ in self._readings)
        surfaces = tuple(surface for surface, _ in self._surfaces)
        reading_indices = {value: index for index, value in enumerate(readings)}
        surface_indices = {value: index for index, value in enumerate(surfaces)}
        try:
            pair_values = tuple(
                value
                for reading, surface, count in self._pairs
                for value in (reading_indices[reading], surface_indices[surface], count)
            )
        except KeyError as error:
            raise ValueError("every pair surface must be present in surface counts") from error
        if self.total_tokens > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("total_tokens does not fit in the table format")
        payload = bytearray(_HEADER.pack(MAGIC, VERSION, 0, 0, self.total_tokens, len(readings), len(surfaces), len(self._pairs)))
        for values in (readings, surfaces):
            encoded = tuple(value.encode("utf-8") for value in values)
            payload.extend(struct.pack(f"<{len(encoded)}I", *(len(value) for value in encoded)) if encoded else b"")
            payload.extend(b"".join(encoded))
        payload.extend(struct.pack(f"<{len(self._readings)}Q", *(count for _, count in self._readings)) if self._readings else b"")
        payload.extend(struct.pack(f"<{len(self._surfaces)}Q", *(count for _, count in self._surfaces)) if self._surfaces else b"")
        payload.extend(struct.pack(f"<{len(pair_values)}I", *pair_values) if pair_values else b"")
        digest = hashlib.sha256(payload).hexdigest()
        Path(path).write_bytes(payload)
        return digest


__all__ = ["CorpusFrequencyTable", "DEFAULT_CORPUS_FREQ_PATH", "MAGIC", "VERSION"]
