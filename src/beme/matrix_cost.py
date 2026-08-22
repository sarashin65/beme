"""Compact IPADIC connection-cost table loading and lookup."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import struct

MAGIC = b"BEMEMAT1"
VERSION = 1
# IPADIC matrix.def costs use the same thousand-unit magnitude as the raw
# connection costs; unlike word costs, their zero point must be retained.
MATRIX_COST_DIVISOR = 1000.0
DEFAULT_MATRIX_PATH = Path(__file__).resolve().parents[2] / "data" / "ipadic_matrix_v1.bin"
_HEADER = struct.Struct("<8sHHI")


def map_matrix_cost(raw_cost: int) -> float:
    """Put raw matrix costs on the existing IPADIC cost scale.

    The existing raw-cost ceiling is used as the scale unit.  Unlike word
    costs, matrix costs are centered around zero, so negative values are
    intentionally preserved rather than clamped.
    """

    return round(raw_cost / MATRIX_COST_DIVISOR, 3)


@dataclass(frozen=True, slots=True)
class MatrixCostTable:
    ids: tuple[int, ...]
    values: tuple[int, ...]
    _indices: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_indices", {value: index for index, value in enumerate(self.ids)})

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MATRIX_PATH) -> "MatrixCostTable":
        with Path(path).open("rb") as stream:
            header = stream.read(_HEADER.size)
            if len(header) != _HEADER.size:
                raise ValueError("matrix table header is truncated")
            magic, version, reserved, count = _HEADER.unpack(header)
            if magic != MAGIC or version != VERSION or reserved != 0:
                raise ValueError("unsupported matrix table format")
            ids_data = stream.read(4 * count)
            if len(ids_data) != 4 * count:
                raise ValueError("matrix table id list is truncated")
            ids = struct.unpack(f"<{count}i", ids_data)
            payload = stream.read(2 * count * count)
            if len(payload) != 2 * count * count or stream.read(1):
                raise ValueError("matrix table payload is truncated or has trailing data")
            values = struct.unpack(f"<{count * count}h", payload)
        if tuple(sorted(set(ids))) != ids:
            raise ValueError("matrix table ids must be sorted and unique")
        return cls(ids, values)

    def lookup(self, previous_right_id: int, current_left_id: int) -> float | None:
        row = self._indices.get(previous_right_id)
        column = self._indices.get(current_left_id)
        if row is None or column is None:
            return None
        return map_matrix_cost(self.values[row * len(self.ids) + column])


__all__ = [
    "DEFAULT_MATRIX_PATH",
    "MAGIC",
    "MATRIX_COST_DIVISOR",
    "MatrixCostTable",
    "VERSION",
    "map_matrix_cost",
]
