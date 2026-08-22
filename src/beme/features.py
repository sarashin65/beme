"""Feature specifications and extraction for path scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from .lattice import LatticeEdge


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Names and identity of a feature extractor specification."""

    id: str
    lexical_features: tuple[str, ...] = ("lex.base_cost",)
    transition_features: tuple[str, ...] = ()

    @classmethod
    def legacy_v1(cls) -> "FeatureSpec":
        return cls(
            "legacy_v1",
            ("lex.base_cost",),
            (
                "trans.bos",
                "trans.pair.noun>particle",
                "trans.pair.proper_noun>particle",
                "trans.pair.pronoun>particle",
                "trans.pair.particle>noun",
                "trans.pair.particle>proper_noun",
                "trans.pair.particle>verb",
                "trans.pair.particle>adjective",
                "trans.pair.verb>particle",
                "trans.pair.adjective>auxiliary",
                "trans.pair.verb>auxiliary",
                "trans.pair.unknown>unknown",
                "trans.pair.unknown>particle",
                "trans.pair.particle>unknown",
                "trans.same_pos",
                "trans.default",
            ),
        )


LEGACY_V1_SPEC = FeatureSpec.legacy_v1()


class FeatureExtractor:
    """Extract the active lexical and transition features for an edge."""

    def __init__(self, spec: FeatureSpec = LEGACY_V1_SPEC) -> None:
        if not isinstance(spec, FeatureSpec):
            raise TypeError("spec must be a FeatureSpec")
        self.spec = spec

    def lexical_features(self, edge: LatticeEdge) -> dict[str, float]:
        if self.spec.id != "legacy_v1":
            raise ValueError(f"unsupported feature specification: {self.spec.id!r}")
        return {"lex.base_cost": edge.token.cost}

    def transition_features(
        self, previous: Hashable | None, edge: LatticeEdge
    ) -> dict[str, float]:
        if self.spec.id != "legacy_v1":
            raise ValueError(f"unsupported feature specification: {self.spec.id!r}")
        current = edge.token.pos
        if previous is None:
            name = "trans.bos"
        else:
            pair = (str(previous), current)
            pair_name = f"trans.pair.{pair[0]}>{pair[1]}"
            if pair_name in self.spec.transition_features:
                name = pair_name
            elif previous == current:
                name = "trans.same_pos"
            else:
                name = "trans.default"
        return {name: 1.0}

    def edge_features(self, edge: LatticeEdge) -> dict[str, float]:
        """Return the lexical features active on one edge."""

        return self.lexical_features(edge)


def legacy_v1_extractor() -> FeatureExtractor:
    return FeatureExtractor(LEGACY_V1_SPEC)
