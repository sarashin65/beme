"""Feature specifications and extraction for path scoring."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import TYPE_CHECKING, Hashable
import unicodedata

from .lattice import LatticeEdge

if TYPE_CHECKING:
    from .lattice import LatticePath
    from .weights import Weights


# Reassociating a sum of n terms costs at most a few ULP; keep a wide margin
# over that while staying far below any real weight mismatch.
_RESIDUAL_TOLERANCE = 1e-9


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

    @classmethod
    def decomposed_v1(cls) -> "FeatureSpec":
        return cls(
            "decomposed_v1",
            (
                "lex.ipadic_cost_mapped", "lex.katakana_surface", "lex.source.overlay",
                "lex.unknown", "lex.literal", "lex.base_cost", "lex.overlay_cost",
                "lex.reading_len", "lex.surface_len", "lex.script.*", "lex.proper_noun",
            ),
            LEGACY_V1_SPEC.transition_features if "LEGACY_V1_SPEC" in globals() else FeatureSpec.legacy_v1().transition_features,
        )


LEGACY_V1_SPEC = FeatureSpec.legacy_v1()


class FeatureExtractor:
    """Extract the active lexical and transition features for an edge."""

    def __init__(self, spec: FeatureSpec = LEGACY_V1_SPEC) -> None:
        if not isinstance(spec, FeatureSpec):
            raise TypeError("spec must be a FeatureSpec")
        self.spec = spec

    def lexical_features(self, edge: LatticeEdge) -> dict[str, float]:
        if self.spec.id == "decomposed_v1":
            token = edge.token
            features: dict[str, float] = {
                "lex.source.overlay": float(token.source == "overlay"),
                "lex.overlay_cost": token.cost if token.source == "overlay" else 0.0,
                "lex.reading_len": float(len(token.reading)),
                "lex.surface_len": float(len(token.surface)),
                "lex.proper_noun": float(token.pos == "proper_noun"),
            }
            normalized_surface = unicodedata.normalize("NFKC", token.surface)
            if normalized_surface and all("\u4e00" <= char <= "\u9fff" for char in normalized_surface):
                script = "kanji"
            elif normalized_surface and all("\u3040" <= char <= "\u309f" for char in normalized_surface):
                script = "hiragana"
            elif normalized_surface and all("\u30a0" <= char <= "\u30ff" for char in normalized_surface):
                script = "katakana"
            else:
                script = "other"
            features[f"lex.script.{script}"] = 1.0
            if token.pos == "unknown":
                features["lex.unknown"] = 1.0
            elif token.pos == "literal":
                features["lex.literal"] = 1.0
            elif token.raw_cost is not None:
                from .ipadic import map_ipadic_cost
                features["lex.ipadic_cost_mapped"] = map_ipadic_cost(token.raw_cost)
                if any("ァ" <= char <= "ヺ" for char in normalized_surface):
                    features["lex.katakana_surface"] = 1.0
            else:
                # v1 dictionaries and hand-written overlays have no raw IPADIC cost.
                features["lex.base_cost"] = token.cost
            return features
        if self.spec.id != "legacy_v1":
            raise ValueError(f"unsupported feature specification: {self.spec.id!r}")
        return {"lex.base_cost": edge.token.cost}

    def transition_features(
        self, previous: Hashable | None, edge: LatticeEdge
    ) -> dict[str, float]:
        if self.spec.id not in {"legacy_v1", "decomposed_v1"}:
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


DECOMPOSED_V1_SPEC = FeatureSpec.decomposed_v1()


def decomposed_v1_extractor() -> FeatureExtractor:
    return FeatureExtractor(DECOMPOSED_V1_SPEC)


def path_features(
    path: LatticePath,
    extractor: FeatureExtractor,
    weights: "Weights | None" = None,
) -> dict[str, float]:
    """Return the feature sum for a path.

    ``LatticePath`` deliberately stores tokens rather than edges.  The phase-1
    extractors are position independent, so edges with reconstructed positions
    are sufficient and keep this API independent of the originating lattice.

    ``weights`` must be the weights that produced ``path.cost``.  They are used
    only to verify the reconstruction; pass the scoring ``LinearScorer``'s
    weights whenever the path was not scored with the default file.
    """

    if not isinstance(extractor, FeatureExtractor):
        raise TypeError("extractor must be a FeatureExtractor")

    features: dict[str, float] = {}
    previous: Hashable | None = None
    position = 0
    for token in path.tokens:
        edge = LatticeEdge(position, position + len(token.reading), token)
        for name, value in extractor.lexical_features(edge).items():
            features[name] = features.get(name, 0.0) + value
        for name, value in extractor.transition_features(previous, edge).items():
            features[name] = features.get(name, 0.0) + value
        previous = extractor_state_key(extractor, edge)
        position = edge.end
    # The scorer accumulates edge and transition terms in path order, while a
    # feature vector necessarily groups equal feature names.  Preserve the
    # legacy scorer's bit-exact total by carrying the rounding residual in its
    # unit-weight lexical feature.  This is only a representation correction:
    # the residual is a reassociation artefact and must stay near one ULP.
    if extractor.spec.id == "legacy_v1" and features:
        if weights is None:
            from .scorer import legacy_v1_weights

            weights = legacy_v1_weights()
        base = weights["lex.base_cost"]
        weighted = sum(weights[name] * value for name, value in features.items())
        residual = path.cost - weighted
        if residual:
            # A residual larger than the accumulated rounding error means the
            # weights do not match the ones that scored the path.  Folding it
            # into lex.base_cost would silently corrupt the vector that drives
            # the phase-2 updates, so refuse instead.
            tolerance = _RESIDUAL_TOLERANCE * max(abs(path.cost), abs(weighted), 1.0)
            if abs(residual) > tolerance:
                raise ValueError(
                    "path cost does not match the supplied weights "
                    f"(cost={path.cost!r}, weighted={weighted!r}); "
                    "pass the weights that scored the path"
                )
            if not base:
                raise ValueError("cannot carry the residual: lex.base_cost weight is zero")
            features["lex.base_cost"] += residual / base
    return features


def extractor_state_key(extractor: FeatureExtractor, edge: LatticeEdge) -> Hashable:
    """Return the state used by the phase-1 extractor for a reconstructed edge."""

    # FeatureExtractor currently has the same state definition as LinearScorer.
    # Keeping this small adapter here avoids making path serialization depend on
    # a scorer instance.
    if extractor.spec.id in {"legacy_v1", "decomposed_v1"}:
        return edge.token.pos
    raise ValueError(f"unsupported feature specification: {extractor.spec.id!r}")


def feature_delta(
    gold: Mapping[str, float], pred: Mapping[str, float]
) -> dict[str, float]:
    """Return ``gold - pred`` with zero terms omitted in sorted key order."""

    delta = {
        name: float(gold.get(name, 0.0)) - float(pred.get(name, 0.0))
        for name in sorted(set(gold) | set(pred))
    }
    return {name: value for name, value in delta.items() if value != 0.0}


__all__ = [
    "FeatureExtractor",
    "FeatureSpec",
    "LEGACY_V1_SPEC",
    "DECOMPOSED_V1_SPEC",
    "decomposed_v1_extractor",
    "feature_delta",
    "legacy_v1_extractor",
    "path_features",
]
