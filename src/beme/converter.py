"""Public hiragana-to-candidate conversion API."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .dictionary import Dictionary
from .lattice import Lattice, Token, build_lattice, k_best_viterbi
from .reranker import ConversionContext, Reranker


@dataclass(frozen=True, slots=True)
class Candidate:
    """One unique surface candidate and its Viterbi path."""

    surface: str
    cost: float
    tokens: tuple[Token, ...]
    score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cost", float(self.cost))
        if not isinstance(self.tokens, tuple):
            object.__setattr__(self, "tokens", tuple(self.tokens))
        score = self.cost if self.score is None else float(self.score)
        if not math.isfinite(score):
            raise ValueError("candidate score must be finite")
        object.__setattr__(self, "score", score)

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "cost": self.cost,
            "score": self.score,
            "tokens": [
                {
                    "reading": token.reading,
                    "surface": token.surface,
                    "pos": token.pos,
                    "cost": token.cost,
                }
                for token in self.tokens
            ],
        }


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """Candidates plus the confidence/cascade decision for one reading."""

    reading: str
    candidates: tuple[Candidate, ...]
    confidence: float | None
    cascade_threshold: float
    needs_rerank: bool
    context: ConversionContext
    reranked: bool

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple):
            object.__setattr__(self, "candidates", tuple(self.candidates))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "cascade_threshold", float(self.cascade_threshold))
        if not isinstance(self.context, ConversionContext):
            raise TypeError("context must be a ConversionContext")

    def to_dict(self) -> dict[str, object]:
        return {
            "reading": self.reading,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "confidence": self.confidence,
            "cascade_threshold": self.cascade_threshold,
            "needs_rerank": self.needs_rerank,
            "context": self.context.to_dict(),
            "reranked": self.reranked,
        }


class Converter:
    """Build lattices and rank their paths with a deterministic K-best beam."""

    def __init__(
        self,
        dictionary: Dictionary,
        beam_width: int = 64,
        reranker: Reranker | None = None,
        rerank_k: int | None = None,
    ) -> None:
        if not isinstance(dictionary, Dictionary):
            raise TypeError("dictionary must be a Dictionary")
        if not isinstance(beam_width, int) or isinstance(beam_width, bool) or beam_width <= 0:
            raise ValueError("beam_width must be a positive integer")
        if rerank_k is not None and (
            not isinstance(rerank_k, int) or isinstance(rerank_k, bool) or rerank_k <= 0
        ):
            raise ValueError("rerank_k must be a positive integer or None")
        self.dictionary = dictionary
        self.beam_width = beam_width
        self.reranker = reranker
        self.rerank_k = rerank_k

    @classmethod
    def from_tsv(
        cls,
        path: str | Path,
        beam_width: int = 64,
        reranker: Reranker | None = None,
        rerank_k: int | None = None,
    ) -> "Converter":
        """Load a strict TSV dictionary and create a converter."""

        return cls(
            Dictionary.from_tsv(path),
            beam_width=beam_width,
            reranker=reranker,
            rerank_k=rerank_k,
        )

    def build_lattice(self, reading: str) -> Lattice:
        return build_lattice(reading, self.dictionary)

    def convert(
        self,
        reading: str,
        k: int = 5,
        cascade_threshold: float = 2.0,
        context: ConversionContext | None = None,
    ) -> ConversionResult:
        """Convert a reading into up to *k* unique, ranked surfaces."""

        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")
        try:
            threshold = float(cascade_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("cascade_threshold must be a finite number") from exc
        if not math.isfinite(threshold):
            raise ValueError("cascade_threshold must be a finite number")
        if context is None:
            context = ConversionContext()
        elif not isinstance(context, ConversionContext):
            raise TypeError("context must be a ConversionContext")

        lattice = self.build_lattice(reading)
        paths = k_best_viterbi(lattice, beam_width=self.beam_width)

        # Terminal paths are already ordered by cost and deterministic tie
        # keys.  Keep the first path for each surface, which is its minimum
        # cost (and its deterministic tie winner).
        by_surface: dict[str, Candidate] = {}
        for path in paths:
            surface = path.surface
            if surface not in by_surface:
                by_surface[surface] = Candidate(surface, path.cost, path.tokens)
        base_candidates = tuple(sorted(by_surface.values(), key=lambda candidate: (candidate.cost, candidate.surface)))
        base_output = base_candidates[:k]
        confidence = None
        if len(base_output) >= 2:
            confidence = base_output[1].cost - base_output[0].cost
        needs_rerank = confidence is not None and confidence < threshold

        reranked = not context.is_empty and self.reranker is not None
        if reranked:
            rerank_candidates = (
                base_candidates
                if self.rerank_k is None
                else base_candidates[: self.rerank_k]
            )
            ranked_candidates = tuple(self.reranker.rerank(rerank_candidates, context))
            if len(ranked_candidates) != len(rerank_candidates) or {
                candidate.surface for candidate in ranked_candidates
            } != {candidate.surface for candidate in rerank_candidates}:
                raise ValueError("reranker must return every input candidate exactly once")
            candidates = ranked_candidates[:k]
        else:
            candidates = base_output
        return ConversionResult(
            reading=reading,
            candidates=candidates,
            confidence=confidence,
            cascade_threshold=threshold,
            needs_rerank=needs_rerank,
            context=context,
            reranked=reranked,
        )


__all__ = ["Candidate", "ConversionResult", "Converter", "Token"]
