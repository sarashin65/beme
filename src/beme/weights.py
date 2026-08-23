"""Versioned JSON persistence for model weights."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import warnings


DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "data" / "weights" / "legacy_v1.json"
WEIGHTS_FORMAT = "beme.weights/1"


@dataclass(frozen=True, slots=True)
class Weights(Mapping[str, float]):
    """A small immutable feature-weight mapping."""

    values: Mapping[str, float]
    default_weight: float = 0.0
    feature_spec: str = "legacy_v1"
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", {name: float(value) for name, value in self.values.items()})
        object.__setattr__(self, "default_weight", float(self.default_weight))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def __getitem__(self, name: str) -> float:
        return float(self.values.get(name, self.default_weight))

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def weights(self) -> Mapping[str, float]:
        return self.values

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible representation, including metadata."""
        document = dict(self.metadata or {})
        document.update(
            {
                "format": WEIGHTS_FORMAT,
                "feature_spec": self.feature_spec,
                "default_weight": self.default_weight,
                "weights": dict(self.values),
            }
        )
        return document

    def dump(self, path: str | Path) -> None:
        """Write deterministic, human-readable JSON weights."""
        target = Path(path)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        feature_spec: str | None = "legacy_v1",
        strict: bool = True,
        dictionary_sha256: str | None = None,
    ) -> "Weights":
        """Load and validate a versioned weights document.

        ``feature_spec`` is the caller's expectation: if the file declares a
        different one, loading fails. Pass ``None`` to accept whatever the
        file declares instead (auto-detection) -- this is for callers such as
        the CLI that dispatch on ``Weights.feature_spec`` after loading and
        have no independent expectation to check the file against; comparing
        it against the value just read back out of the same file would never
        fire (issue #9 Phase 1 review, L1).
        """
        source = Path(path)
        with source.open(encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, dict):
            raise ValueError(f"weights file must contain a JSON object: {source}")
        if document.get("format") != WEIGHTS_FORMAT:
            raise ValueError(f"unsupported weights format: {document.get('format')!r}")
        actual_spec = document.get("feature_spec")
        if feature_spec is None:
            if not isinstance(actual_spec, str):
                raise ValueError(f"weights file does not declare a feature_spec: {source}")
            feature_spec = actual_spec
        elif actual_spec != feature_spec:
            raise ValueError(
                f"feature_spec mismatch: file has {actual_spec!r}, expected {feature_spec!r}"
            )
        if "weights_path" in document:
            raise NotImplementedError("external weights_path is reserved but not implemented")
        values = document.get("weights", {})
        if not isinstance(values, dict):
            raise ValueError("weights must be a JSON object")
        from .features import resolve_spec

        spec = resolve_spec(feature_spec)
        known = set(spec.lexical_features) | set(spec.transition_features)
        unknown = sorted(set(values) - known)
        if unknown and strict:
            raise ValueError(f"unknown feature name(s): {', '.join(unknown)}")
        if unknown:
            warnings.warn(f"ignoring unknown feature name(s): {', '.join(unknown)}", UserWarning)
        try:
            default_weight = float(document.get("default_weight", 0.0))
            clean_values = {name: float(value) for name, value in values.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("weights and default_weight must be numbers") from exc
        if dictionary_sha256 and document.get("dictionary_sha256") not in (None, dictionary_sha256):
            warnings.warn("dictionary_sha256 does not match the active dictionary", UserWarning)
        metadata = {key: value for key, value in document.items() if key not in {"weights", "default_weight"}}
        return cls(clean_values, default_weight, feature_spec, metadata)


def load(path: str | Path = DEFAULT_WEIGHTS_PATH, *, strict: bool = True) -> Weights:
    return Weights.load(path, strict=strict)


def dump(weights: Weights, path: str | Path) -> None:
    weights.dump(path)


__all__ = ["DEFAULT_WEIGHTS_PATH", "WEIGHTS_FORMAT", "Weights", "dump", "load"]
