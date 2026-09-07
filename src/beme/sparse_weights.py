"""Sparse lexical weights with pair-surface-reading backoff."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SparseLexicalWeights:
    """Sparse lexical weights with three-level backoff: pair → surface → reading → 0.0.

    Each level is a dict[str, float] mapping a key to its learned weight.
    The backoff chain is:
        1. w_{r,s} (reading, surface pair)  — most specific
        2. w_s     (surface only)            — intermediate
        3. w_r     (reading only)            — most general
        4. 0.0                               — default

    During training, only pair and surface levels are updated (see
    training._update_sparse_weights). The reading level is kept for
    backoff reference only and is not updated during learning, to
    avoid cancellation between different surfaces of the same reading
    (homophone ambiguity). This means the effective backoff depth is
    2 levels during training and 3 levels at inference time.
    """

    pair_weights: dict[str, float]
    surface_weights: dict[str, float]
    reading_weights: dict[str, float]

    @classmethod
    def empty(cls) -> SparseLexicalWeights:
        return cls({}, {}, {})

    def copy(self) -> SparseLexicalWeights:
        return SparseLexicalWeights(
            dict(self.pair_weights),
            dict(self.surface_weights),
            dict(self.reading_weights),
        )

    @staticmethod
    def _pair_key(reading: str, surface: str) -> str:
        return f"{reading}\t{surface}"

    def get(self, reading: str, surface: str) -> float:
        """Return the backoff-resolved weight for the given (reading, surface)."""
        pair_key = self._pair_key(reading, surface)
        if pair_key in self.pair_weights:
            return self.pair_weights[pair_key]
        if surface in self.surface_weights:
            return self.surface_weights[surface]
        if reading in self.reading_weights:
            return self.reading_weights[reading]
        return 0.0

    def get_pair(self, reading: str, surface: str) -> float:
        return self.pair_weights.get(self._pair_key(reading, surface), 0.0)

    def get_surface(self, surface: str) -> float:
        return self.surface_weights.get(surface, 0.0)

    def get_reading(self, reading: str) -> float:
        return self.reading_weights.get(reading, 0.0)

    def set_pair(self, reading: str, surface: str, value: float) -> None:
        key = self._pair_key(reading, surface)
        if abs(value) < 1e-12:
            self.pair_weights.pop(key, None)
        else:
            self.pair_weights[key] = value

    def set_surface(self, surface: str, value: float) -> None:
        if abs(value) < 1e-12:
            self.surface_weights.pop(surface, None)
        else:
            self.surface_weights[surface] = value

    def set_reading(self, reading: str, value: float) -> None:
        if abs(value) < 1e-12:
            self.reading_weights.pop(reading, None)
        else:
            self.reading_weights[reading] = value

    def update_pair(self, reading: str, surface: str, delta: float) -> None:
        current = self.get_pair(reading, surface)
        self.set_pair(reading, surface, current + delta)

    def update_surface(self, surface: str, delta: float) -> None:
        current = self.get_surface(surface)
        self.set_surface(surface, current + delta)

    def update_reading(self, reading: str, delta: float) -> None:
        current = self.get_reading(reading)
        self.set_reading(reading, current + delta)

    def apply_l2(self, rate: float) -> None:
        """Apply L2 regularization: w -= rate * w for all weights."""
        decay = 1.0 - rate
        for key in list(self.pair_weights):
            new_val = self.pair_weights[key] * decay
            if abs(new_val) < 1e-12:
                del self.pair_weights[key]
            else:
                self.pair_weights[key] = new_val
        for key in list(self.surface_weights):
            new_val = self.surface_weights[key] * decay
            if abs(new_val) < 1e-12:
                del self.surface_weights[key]
            else:
                self.surface_weights[key] = new_val
        for key in list(self.reading_weights):
            new_val = self.reading_weights[key] * decay
            if abs(new_val) < 1e-12:
                del self.reading_weights[key]
            else:
                self.reading_weights[key] = new_val

    @property
    def num_nonzero(self) -> dict[str, int]:
        return {
            "pair": len(self.pair_weights),
            "surface": len(self.surface_weights),
            "reading": len(self.reading_weights),
        }

    # ── I/O ──────────────────────────────────────────────────────────

    SIDECAR_MAGIC = b"BEMESLW1"
    SIDECAR_HEADER_FMT = "=8sIIII"  # magic(8) + version(4) + n_pairs(4) + n_surfaces(4) + n_readings(4)
    SIDECAR_ENTRY_FMT = "=f"       # float32 per value

    def dump(self, path: Path) -> None:
        """Write JSON manifest + binary sidecar.

        The JSON manifest holds metadata; the sidecar holds the actual values
        as a flat float32 array.  This keeps the JSON readable while the
        binary part is compact and deterministic.
        """
        manifest_path = path.with_suffix(".sparse.json")
        sidecar_path = path.with_suffix(".sparse.bin")

        # Build ordered index for each level
        pair_keys = sorted(self.pair_weights)
        surface_keys = sorted(self.surface_weights)
        reading_keys = sorted(self.reading_weights)

        # Serialize values to binary sidecar
        values = (
            [self.pair_weights[k] for k in pair_keys]
            + [self.surface_weights[k] for k in surface_keys]
            + [self.reading_weights[k] for k in reading_keys]
        )
        header = struct.pack(
            self.SIDECAR_HEADER_FMT,
            self.SIDECAR_MAGIC,
            1,  # version
            len(pair_keys),
            len(surface_keys),
            len(reading_keys),
        )
        payload = struct.pack(f"={len(values)}f", *values)
        sidecar_path.write_bytes(header + payload)

        # Write JSON manifest
        manifest = {
            "format": "beme.sparse_weights/1",
            "sidecar": sidecar_path.name,
            "sidecar_sha256": _sha256(sidecar_path),
            "indices": {
                "pair": pair_keys,
                "surface": surface_keys,
                "reading": reading_keys,
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> SparseLexicalWeights:
        """Load from JSON manifest + binary sidecar.

        Accepts either the manifest path (``.sparse.json``) or a base path.
        """
        manifest_path = path if path.suffix == ".sparse.json" else path.with_suffix(".sparse.json")
        sidecar_path = manifest_path.with_suffix(".bin")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "beme.sparse_weights/1":
            raise ValueError(f"unsupported format: {manifest.get('format')!r}")

        # Verify sidecar integrity
        expected_sha = manifest.get("sidecar_sha256")
        if expected_sha and _sha256(sidecar_path) != expected_sha:
            raise ValueError("sidecar SHA-256 mismatch")

        # Parse binary sidecar
        data = sidecar_path.read_bytes()
        header_size = struct.calcsize(cls.SIDECAR_HEADER_FMT)
        magic, version, n_pairs, n_surfaces, n_readings = struct.unpack(
            cls.SIDECAR_HEADER_FMT, data[:header_size]
        )
        if magic != cls.SIDECAR_MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        if version != 1:
            raise ValueError(f"unsupported sidecar version: {version}")

        n_values = n_pairs + n_surfaces + n_readings
        expected_size = header_size + n_values * struct.calcsize(cls.SIDECAR_ENTRY_FMT)
        if len(data) != expected_size:
            raise ValueError(f"sidecar size mismatch: expected {expected_size}, got {len(data)}")

        values = struct.unpack(f"={n_values}f", data[header_size:])

        # Reconstruct indices
        pair_keys = manifest["indices"]["pair"]
        surface_keys = manifest["indices"]["surface"]
        reading_keys = manifest["indices"]["reading"]

        pair_weights = dict(zip(pair_keys, values[:n_pairs]))
        surface_weights = dict(zip(surface_keys, values[n_pairs:n_pairs + n_surfaces]))
        reading_weights = dict(zip(reading_keys, values[n_pairs + n_surfaces:]))

        # Prune zeros
        pair_weights = {k: v for k, v in pair_weights.items() if abs(v) >= 1e-12}
        surface_weights = {k: v for k, v in surface_weights.items() if abs(v) >= 1e-12}
        reading_weights = {k: v for k, v in reading_weights.items() if abs(v) >= 1e-12}

        return cls(pair_weights, surface_weights, reading_weights)

    # ── dict conversion (for training intermediate state) ─────────────

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "beme.sparse_weights/1",
            "pair_weights": dict(self.pair_weights),
            "surface_weights": dict(self.surface_weights),
            "reading_weights": dict(self.reading_weights),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SparseLexicalWeights:
        return cls(
            dict(data.get("pair_weights", {})),
            dict(data.get("surface_weights", {})),
            dict(data.get("reading_weights", {})),
        )


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["SparseLexicalWeights"]
