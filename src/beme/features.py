"""Feature specifications and extraction for path scoring."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping
from functools import lru_cache
import math
from typing import TYPE_CHECKING, Hashable, NamedTuple
import unicodedata

from .corpus_freq import CORPUS_LOGPROB_TEMPERATURE
from .lattice import LatticeEdge

if TYPE_CHECKING:
    from .corpus_freq import CorpusFrequencyTable
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
    transition_state: tuple[str, ...] = ("pos",)

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

    def __init__(
        self,
        spec: FeatureSpec = LEGACY_V1_SPEC,
        reading_conditional: Mapping[str, tuple[float, int]] | None = None,
        *,
        corpus_frequency: "CorpusFrequencyTable | None" = None,
    ) -> None:
        if not isinstance(spec, FeatureSpec):
            raise TypeError("spec must be a FeatureSpec")
        self.spec = spec
        self.reading_conditional = reading_conditional
        self.corpus_frequency = corpus_frequency
        # Whether the active spec declares corpus features.  The 340k-pair
        # table itself is resolved lazily inside lexical_features (matrix
        # style), never loaded for the other specifications.
        self._needs_corpus = bool(set(spec.lexical_features) & set(CORPUS_LEXICAL_FEATURES))

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
            if {"lex.reading_cond_logprob", "lex.reading_candidate_count"} & set(self.spec.lexical_features):
                if self.reading_conditional is None:
                    raise ValueError("reading conditional table is required for this feature specification")
                log_partition, count = self.reading_conditional.get(token.reading, (0.0, 0))
                if token.raw_cost is not None:
                    from .ipadic import READING_CONDITIONAL_TEMPERATURE

                    features["lex.reading_cond_logprob"] = (
                        token.raw_cost / READING_CONDITIONAL_TEMPERATURE + log_partition
                    )
                features["lex.reading_candidate_count"] = math.log1p(count)
            if self._needs_corpus:
                # Lazy singleton (matrix style): resolve the table only on
                # first use, and only for specs that declare corpus features.
                table = self.corpus_frequency
                if table is None:
                    try:
                        table = _corpus_table()
                    except FileNotFoundError as error:
                        # An explicit table was not injected and the default
                        # file does not exist: fail loudly instead of scoring
                        # every edge as unseen (issue #19 F2).
                        raise ValueError(
                            "corpus frequency table is required for this feature specification"
                        ) from error
                pair_count = table.pair_count(token.reading, token.surface)
                reading_count = table.reading_count(token.reading)
                surface_count = table.surface_count(token.surface)
                # Values are negative log probabilities (cost-oriented, like
                # every other lexical feature), not log probabilities.
                if pair_count > 0 and reading_count > 0:
                    features["lex.corpus_reading_cond_logprob"] = (
                        -math.log(pair_count / reading_count) / CORPUS_LOGPROB_TEMPERATURE
                    )
                    features["lex.corpus_unseen"] = 0.0
                else:
                    features["lex.corpus_reading_cond_logprob"] = 0.0
                    features["lex.corpus_unseen"] = 1.0
                if surface_count > 0 and table.total_tokens > 0:
                    features["lex.corpus_unigram_logprob"] = (
                        -math.log(surface_count / table.total_tokens) / CORPUS_LOGPROB_TEMPERATURE
                    )
                else:
                    features["lex.corpus_unigram_logprob"] = 0.0
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
        previous_pos = previous.pos if isinstance(previous, TransitionState) else previous
        if previous is None:
            name = "trans.bos"
        else:
            pair = (str(previous_pos), current)
            pair_name = f"trans.pair.{pair[0]}>{pair[1]}"
            if pair_name in self.spec.transition_features:
                name = pair_name
            elif previous_pos == current:
                name = "trans.same_pos"
            else:
                name = "trans.default"
        features = {name: 1.0} if name in self.spec.transition_features else {}
        if "trans.matrix_cost" in self.spec.transition_features:
            matrix = _matrix_table()
            previous_right_id = 0 if previous is None else previous.right_id
            matrix_value = matrix.lookup(previous_right_id, edge.token.left_id)
            if matrix_value is not None:
                features["trans.matrix_cost"] = matrix_value
        return features

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
        base.transition_state,
    )


class TransitionState(NamedTuple):
    pos: str
    right_id: int


@lru_cache(maxsize=1)
def _matrix_table():
    from .matrix_cost import DEFAULT_MATRIX_PATH, MatrixCostTable

    return MatrixCostTable.load(DEFAULT_MATRIX_PATH)


@lru_cache(maxsize=1)
def _corpus_table():
    from .corpus_freq import DEFAULT_CORPUS_FREQ_PATH, CorpusFrequencyTable

    return CorpusFrequencyTable.load(DEFAULT_CORPUS_FREQ_PATH)


MATRIX_V1_SPEC = FeatureSpec(
    "decomposed_v1+matrix_v1",
    DECOMPOSED_V1_SPEC.lexical_features,
    (*DECOMPOSED_V1_SPEC.transition_features, "trans.matrix_cost"),
    DECOMPOSED_V1_SPEC.family,
    ("pos", "right_id"),
)
MATRIX_V1_POS_PAIRS_SPEC = FeatureSpec(
    "decomposed_v1+matrix_v1-pos_pairs",
    MATRIX_V1_SPEC.lexical_features,
    tuple(name for name in MATRIX_V1_SPEC.transition_features
          if not name.startswith("trans.pair.")),
    MATRIX_V1_SPEC.family,
    MATRIX_V1_SPEC.transition_state,
)

FREQ_LEXICAL_FEATURES = (
    "lex.reading_cond_logprob", "lex.reading_candidate_count",
)
FREQ_V1_SPEC = FeatureSpec(
    "decomposed_v1+freq_v1",
    (*DECOMPOSED_V1_SPEC.lexical_features, *FREQ_LEXICAL_FEATURES),
    DECOMPOSED_V1_SPEC.transition_features,
    DECOMPOSED_V1_SPEC.family,
)
FREQ_V1_NO_IPADIC_SPEC = FeatureSpec(
    "decomposed_v1+freq_v1-no_ipadic_cost",
    tuple(name for name in FREQ_V1_SPEC.lexical_features if name != "lex.ipadic_cost_mapped"),
    FREQ_V1_SPEC.transition_features,
    FREQ_V1_SPEC.family,
)
MATRIX_FREQ_V1_SPEC = FeatureSpec(
    "decomposed_v1+matrix_v1+freq_v1",
    FREQ_V1_SPEC.lexical_features,
    MATRIX_V1_SPEC.transition_features,
    MATRIX_V1_SPEC.family,
    MATRIX_V1_SPEC.transition_state,
)

# Corpus-frequency lexical features (issue #19 F2).  Values are negative log
# probabilities divided by CORPUS_LOGPROB_TEMPERATURE, i.e. cost-oriented like
# every other lexical feature; larger means worse.  No smoothing: the constant
# offset for unseen pairs is carried by the learned weight of
# lex.corpus_unseen.
CORPUS_LEXICAL_FEATURES = (
    "lex.corpus_reading_cond_logprob",
    "lex.corpus_unigram_logprob",
    "lex.corpus_unseen",
)
# C1: the issue #19 comment's condition (A0 + 3): 14 lexical + 16 transition
# + 3 corpus = 33 parameters, kept for the A0/A8 contrast.
CORPUS_V1_A0_SPEC = FeatureSpec(
    "decomposed_v1+corpus_v1",
    (*DECOMPOSED_V1_SPEC.lexical_features, *CORPUS_LEXICAL_FEATURES),
    DECOMPOSED_V1_SPEC.transition_features,
    DECOMPOSED_V1_SPEC.family,
)


_ABLATION_REMOVALS = {
    "decomposed_v1-lex_only": set(DECOMPOSED_V1_SPEC.transition_features),
    "decomposed_v1-trans_only": set(DECOMPOSED_V1_SPEC.lexical_features),
    "decomposed_v1-no_ipadic_cost": {"lex.ipadic_cost_mapped"},
    "decomposed_v1-no_source": {"lex.source.overlay", "lex.overlay_cost"},
    "decomposed_v1-no_katakana": {"lex.katakana_surface"},
    "decomposed_v1-no_script": {
        *(f"lex.script.{category}" for category in _SCRIPT_CATEGORIES),
    },
    "decomposed_v1-no_script-surface_len": {
        *(f"lex.script.{category}" for category in _SCRIPT_CATEGORIES),
        "lex.surface_len",
    },
    "decomposed_v1-no_shape": {
        *(f"lex.script.{category}" for category in _SCRIPT_CATEGORIES),
        "lex.surface_len", "lex.proper_noun",
    },
}
_A8_BASE = _spec_without(
    DECOMPOSED_V1_SPEC, "decomposed_v1-no_script-surface_len",
    _ABLATION_REMOVALS["decomposed_v1-no_script-surface_len"],
)
# C0: the current base is A8 (issue #17 / commit a4b690b), so the corpus
# features stack on A8: 9 lexical + 16 transition + 3 corpus = 28 parameters.
CORPUS_V1_A8_SPEC = FeatureSpec(
    "decomposed_v1-no_script-surface_len+corpus_v1",
    (*_A8_BASE.lexical_features, *CORPUS_LEXICAL_FEATURES),
    _A8_BASE.transition_features,
    DECOMPOSED_V1_SPEC.family,
)
_SPEC_REGISTRY = {
    "legacy_v1": LEGACY_V1_SPEC,
    "decomposed_v1": DECOMPOSED_V1_SPEC,
    **{
        spec_id: _spec_without(DECOMPOSED_V1_SPEC, spec_id, removed)
        for spec_id, removed in _ABLATION_REMOVALS.items()
    },
    MATRIX_V1_SPEC.id: MATRIX_V1_SPEC,
    MATRIX_V1_POS_PAIRS_SPEC.id: MATRIX_V1_POS_PAIRS_SPEC,
    FREQ_V1_SPEC.id: FREQ_V1_SPEC,
    FREQ_V1_NO_IPADIC_SPEC.id: FREQ_V1_NO_IPADIC_SPEC,
    MATRIX_FREQ_V1_SPEC.id: MATRIX_FREQ_V1_SPEC,
    CORPUS_V1_A8_SPEC.id: CORPUS_V1_A8_SPEC,
    CORPUS_V1_A0_SPEC.id: CORPUS_V1_A0_SPEC,
}


def resolve_spec(spec_id: str) -> FeatureSpec:
    try:
        return _SPEC_REGISTRY[spec_id]
    except KeyError as exc:
        raise ValueError(f"unsupported feature specification: {spec_id!r}") from exc


def extractor_for(
    spec_id: str,
    reading_conditional: Mapping[str, tuple[float, int]] | None = None,
    *,
    corpus_frequency: "CorpusFrequencyTable | None" = None,
) -> FeatureExtractor:
    return FeatureExtractor(
        resolve_spec(spec_id), reading_conditional, corpus_frequency=corpus_frequency
    )


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
    state_key = state_key_function(extractor.spec)
    for token in path.tokens:
        edge = LatticeEdge(position, position + len(token.reading), token)
        for name, value in extractor.lexical_features(edge).items():
            features[name] = features.get(name, 0.0) + value
        for name, value in extractor.transition_features(previous, edge).items():
            features[name] = features.get(name, 0.0) + value
        previous = state_key(edge)
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


def _pos_state_key(edge: LatticeEdge) -> Hashable:
    return edge.token.pos


def _pos_right_id_state_key(edge: LatticeEdge) -> Hashable:
    return TransitionState(edge.token.pos, edge.token.right_id)


def state_key_function(spec: FeatureSpec) -> Callable[[LatticeEdge], Hashable]:
    """Resolve a specification's state definition to a callable, once.

    ``state_key`` is called several times per edge from the k-best inner loop
    (``lattice.k_best_viterbi``), so the specification must be inspected when
    the scorer is built rather than on every call.
    """

    if spec.transition_state == ("pos",):
        return _pos_state_key
    if spec.transition_state == ("pos", "right_id"):
        return _pos_right_id_state_key
    raise ValueError(f"unsupported feature specification: {spec.id!r}")


def extractor_state_key(extractor: FeatureExtractor, edge: LatticeEdge) -> Hashable:
    """Return the state used by the phase-1 extractor for a reconstructed edge."""

    # FeatureExtractor currently has the same state definition as LinearScorer.
    # Keeping this small adapter here avoids making path serialization depend on
    # a scorer instance.
    return state_key_function(extractor.spec)(edge)


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
    "MATRIX_V1_SPEC",
    "MATRIX_V1_POS_PAIRS_SPEC",
    "FREQ_V1_SPEC",
    "FREQ_V1_NO_IPADIC_SPEC",
    "MATRIX_FREQ_V1_SPEC",
    "TransitionState",
    "decomposed_v1_extractor",
    "extractor_for",
    "feature_delta",
    "legacy_v1_extractor",
    "path_features",
    "registered_spec_ids",
    "resolve_spec",
    "state_key_function",
]
