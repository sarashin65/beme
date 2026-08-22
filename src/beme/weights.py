"""In-memory model weights used by the scorer."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Weights(Mapping[str, float]):
    """A small immutable feature-weight mapping."""

    values: Mapping[str, float]
    default_weight: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", {name: float(value) for name, value in self.values.items()})
        object.__setattr__(self, "default_weight", float(self.default_weight))

    def __getitem__(self, name: str) -> float:
        return float(self.values.get(name, self.default_weight))

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def weights(self) -> Mapping[str, float]:
        return self.values
