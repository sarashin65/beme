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

# The surface-script family of features. Kept as one tuple so the feature
# names declared by FeatureSpec.decomposed_v1() and the names emitted by
# FeatureExtractor.lexical_features() cannot drift apart the way the single
# glob string "lex.script.*" once did (see issue #9 Phase 1 review, M1):
# Weights.load() only recognizes literal feature names, so any name it does
# not see here it rejects outright.
_SCRIPT_CATEGORIES = ("kanji", "hiragana", "katakana", "other")


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Names and identity of a feature extractor specification."""

    id: str
    lexical_features: tuple[str, ...] = ("lex.base_cost",)
    transition_features: tuple[str, ...] = ()
    family: str = "legacy_v1"

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
            "legacy_v1",
        )

    @classmethod
    def decomposed_v1(cls) -> "FeatureSpec":
        return cls(
            "decomposed_v1",
            (
                "lex.ipadic_cost_mapped", "lex.katakana_surface", "lex.source.overlay",
                "lex.unknown", "lex.literal", "lex.base_cost", "lex.overlay_cost",
                "lex.reading_len", "lex.surface_len",
                *(f"lex.script.{category}" for category in _SCRIPT_CATEGORIES),
                "lex.proper_noun",
            ),
            LEGACY_V1_SPEC.transition_features if "LEGACY_V1_SPEC" in globals() else FeatureSpec.legacy_v1().transition_features,
            "decomposed_v1",
        )


LEGACY_V1_SPEC = FeatureSpec.legacy_v1()


class FeatureExtractor:
    """Extract the active lexical and transition features for an edge."""

    def __init__(self, spec: FeatureSpec = LEGACY_V1_SPEC) -> None:
        if not isinstance(spec, FeatureSpec):
            raise TypeError("spec must be a FeatureSpec")
        self.spec = spec

    def lexical_features(self, edge: LatticeEdge) -> dict[str, float]:
        if self.spec.family == "decomposed_v1":
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
            assert script in _SCRIPT_CATEGORIES
            features[f"lex.script.{script}"] = 1.0
            if token.pos == "unknown":
                features["lex.unknown"] = 1.0
            elif token.pos == "literal":
                features["lex.literal"] = 1.0
            elif token.raw_cost is not None:
                # Do not recompute the IPADIC cost mapping here: it would
                # silently ignore whatever --raw-cost-floor/--raw-cost-ceiling/
                # --beme-cost-floor/--beme-cost-ceiling the dictionary was
                # actually built with (scripts/build_ipadic_dictionary.py),
                # and any later scripts/apply_cost_learning.py adjustment
                # (issue #9 Phase 1 review, M4). token.cost is the one place
                # those parameters and adjustments actually land, so derive
                # the mapped-cost feature from it instead. It only needs
                # subtracting back out the katakana-surface penalty that the
                # same build step folded in, so the two decomposed features
                # stay orthogonal and reconstruction stays exact. Use the
                # dictionary build's own katakana predicate (issue #9 Phase 1
                # review, L3) rather than a narrower ad-hoc range check.
                from .ipadic import KATAKANA_SURFACE_COST_PENALTY, contains_katakana_surface

                is_katakana_surface = contains_katakana_surface(token.surface)
                features["lex.ipadic_cost_mapped"] = token.cost - (
                    KATAKANA_SURFACE_COST_PENALTY if is_katakana_surface else 0.0
                )
                if is_katakana_surface:
                    features["lex.katakana_surface"] = 1.0
            else:
                # v1 dictionaries and hand-written overlays have no raw IPADIC cost.
                features["lex.base_cost"] = token.cost
            return {name: value for name, value in features.items()
                    if name in self.spec.lexical_features}
        if self.spec.family != "legacy_v1":
            raise ValueError(f"unsupported feature specification: {self.spec.id!r}")
        features = {"lex.base_cost": edge.token.cost}
        return {name: value for name, value in features.items()
                if name in self.spec.lexical_features}

    def transition_features(
        self, previous: Hashable | None, edge: LatticeEdge
    ) -> dict[str, float]:
        if self.spec.family not in {"legacy_v1", "decomposed_v1"}:
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
        return {name: 1.0} if name in self.spec.transition_features else {}

    def edge_features(self, edge: LatticeEdge) -> dict[str, float]:
        """Return the lexical features active on one edge."""

        return self.lexical_features(edge)


def legacy_v1_extractor() -> FeatureExtractor:
    return extractor_for("legacy_v1")


DECOMPOSED_V1_SPEC = FeatureSpec.decomposed_v1()


def _spec_without(base: FeatureSpec, spec_id: str, removed: set[str]) -> FeatureSpec:
    return FeatureSpec(
        spec_id,
        tuple(name for name in base.lexical_features if name not in removed),
        tuple(name for name in base.transition_features if name not in removed),
        base.family,
    )


_ABLATION_REMOVALS = {
    "decomposed_v1-lex_only": set(DECOMPOSED_V1_SPEC.transition_features),
    "decomposed_v1-trans_only": set(DECOMPOSED_V1_SPEC.lexical_features),
    "decomposed_v1-no_ipadic_cost": {"lex.ipadic_cost_mapped"},
    "decomposed_v1-no_source": {"lex.source.overlay", "lex.overlay_cost"},
    "decomposed_v1-no_katakana": {"lex.katakana_surface"},
    "decomposed_v1-no_shape": {
        *(f"lex.script.{category}" for category in _SCRIPT_CATEGORIES),
        "lex.surface_len", "lex.proper_noun",
    },
}
_SPEC_REGISTRY = {
    "legacy_v1": LEGACY_V1_SPEC,
    "decomposed_v1": DECOMPOSED_V1_SPEC,
    **{
        spec_id: _spec_without(DECOMPOSED_V1_SPEC, spec_id, removed)
        for spec_id, removed in _ABLATION_REMOVALS.items()
    },
}


def resolve_spec(spec_id: str) -> FeatureSpec:
    try:
        return _SPEC_REGISTRY[spec_id]
    except KeyError as exc:
        raise ValueError(f"unsupported feature specification: {spec_id!r}") from exc


def extractor_for(spec_id: str) -> FeatureExtractor:
    return FeatureExtractor(resolve_spec(spec_id))


def registered_spec_ids() -> tuple[str, ...]:
    return tuple(_SPEC_REGISTRY)


def decomposed_v1_extractor() -> FeatureExtractor:
    return extractor_for("decomposed_v1")


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
    if extractor.spec.family == "legacy_v1" and features:
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
    if extractor.spec.family in {"legacy_v1", "decomposed_v1"}:
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
    "extractor_for",
    "feature_delta",
    "legacy_v1_extractor",
    "path_features",
    "registered_spec_ids",
    "resolve_spec",
]
