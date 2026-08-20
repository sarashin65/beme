"""Phase 2 context reranking contract and deterministic mock implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .converter import Candidate


@dataclass(frozen=True, slots=True)
class ConversionContext:
    """Text surrounding the reading currently being converted."""

    preceding: str = ""
    following: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.preceding, str):
            raise TypeError("preceding context must be a string")
        if not isinstance(self.following, str):
            raise TypeError("following context must be a string")

    @property
    def is_empty(self) -> bool:
        return not self.preceding and not self.following

    def to_dict(self) -> dict[str, str]:
        return {"preceding": self.preceding, "following": self.following}


class Reranker(Protocol):
    """Contract for a Phase 2 ranker; lower candidate scores rank first."""

    def rerank(
        self,
        candidates: Sequence[Candidate],
        context: ConversionContext,
    ) -> tuple[Candidate, ...]:
        """Return every input candidate once, ordered and scored."""


_DEFAULT_CONTEXT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "考証": ("歴史", "史料", "文献", "研究", "検証"),
    "交渉": ("商談", "取引", "契約", "条件", "折衝"),
    "校章": ("学校", "制服", "徽章"),
    "高尚": ("品格", "精神", "趣味"),
    "工廠": ("軍需", "兵器", "工場"),
    "鉱床": ("地質", "鉱山", "資源"),
}


class MockContextReranker:
    """Deterministic keyword-overlap reranker used before an LLM is connected.

    Each matching keyword subtracts ``keyword_bonus`` from the Phase 1 path
    cost.  This keeps the mock explainable while exercising the Phase 2 input
    and output contract.  Lower scores rank first.
    """

    def __init__(
        self,
        keyword_bonus: float = 1.0,
        keywords: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        if isinstance(keyword_bonus, bool) or not isinstance(keyword_bonus, (int, float)):
            raise TypeError("keyword_bonus must be a number")
        if not math.isfinite(keyword_bonus) or keyword_bonus < 0:
            raise ValueError("keyword_bonus must be finite and non-negative")
        source = _DEFAULT_CONTEXT_KEYWORDS if keywords is None else keywords
        normalized: dict[str, tuple[str, ...]] = {}
        for surface, surface_keywords in source.items():
            if not isinstance(surface, str) or not surface:
                raise ValueError("keyword surfaces must be non-empty strings")
            values = tuple(surface_keywords)
            if any(not isinstance(keyword, str) or not keyword for keyword in values):
                raise ValueError("context keywords must be non-empty strings")
            normalized[surface] = values
        self.keyword_bonus = float(keyword_bonus)
        self.keywords = normalized

    def contextual_bonus(self, candidate: Candidate, context: ConversionContext) -> float:
        """Return the keyword-match bonus for one candidate and context."""

        matches = 0
        for keyword in self.keywords.get(candidate.surface, ()):
            matches += context.preceding.count(keyword)
            matches += context.following.count(keyword)
        return self.keyword_bonus * matches

    def score(self, candidate: Candidate, context: ConversionContext) -> float:
        """Return adjusted ranking cost; lower is better."""

        return candidate.cost - self.contextual_bonus(candidate, context)

    def rerank(
        self,
        candidates: Sequence[Candidate],
        context: ConversionContext,
    ) -> tuple[Candidate, ...]:
        if not isinstance(context, ConversionContext):
            raise TypeError("context must be a ConversionContext")
        scored = tuple(replace(candidate, score=self.score(candidate, context)) for candidate in candidates)
        return tuple(sorted(scored, key=lambda candidate: (candidate.score, candidate.cost, candidate.surface)))


__all__ = ["ConversionContext", "MockContextReranker", "Reranker"]
