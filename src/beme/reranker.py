"""Phase 2 context reranking contract and mock/LLM implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, TYPE_CHECKING, Protocol

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

class LLMRerankerUnavailable(RuntimeError):
    """Raised when the local llama.cpp reranker cannot be loaded."""


DEFAULT_LLM_MODEL_PATH = (
    Path.home() / ".cache" / "beme" / "models" / "Qwen2.5-0.5B.Q4_K_M.gguf"
)
DEFAULT_LLM_WEIGHT = 1.0
DEFAULT_VITERBI_WEIGHT = 1.0
DEFAULT_LLM_CONTEXT_SIZE = 256


class _LlamaBackend(Protocol):
    """Small llama-cpp-python surface used by the reranker and test doubles."""

    n_tokens: int
    scores: Any

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> list[int]: ...

    def token_bos(self) -> int: ...

    def token_eos(self) -> int: ...

    def reset(self) -> None: ...

    def eval(self, tokens: Sequence[int]) -> None: ...

    def save_state(self) -> Any: ...

    def load_state(self, state: Any) -> None: ...

    def logits_to_logprobs(self, logits: Any, axis: int = -1) -> Any: ...


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


class LLMLogitsReranker:
    """Rank candidates by Qwen continuation likelihood plus Viterbi cost.

    The confirmed preceding text is evaluated once.  Its llama.cpp state,
    including the KV cache, is restored before each candidate continuation.
    When right context is available, the continuation is ``candidate +
    following`` so its likelihood can distinguish candidates that precede the
    same text.  Mean token log-likelihood prevents token-count bias.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_LLM_MODEL_PATH,
        *,
        llm_weight: float = DEFAULT_LLM_WEIGHT,
        viterbi_weight: float = DEFAULT_VITERBI_WEIGHT,
        context_size: int = DEFAULT_LLM_CONTEXT_SIZE,
        model: _LlamaBackend | None = None,
    ) -> None:
        self.llm_weight = self._weight("llm_weight", llm_weight)
        self.viterbi_weight = self._weight("viterbi_weight", viterbi_weight)
        if self.llm_weight == 0 and self.viterbi_weight == 0:
            raise ValueError("at least one reranker weight must be positive")
        if isinstance(context_size, bool) or not isinstance(context_size, int) or context_size < 2:
            raise ValueError("context_size must be an integer of at least 2")
        self.context_size = context_size
        self.model_path = Path(model_path).expanduser()
        self._model = model if model is not None else self._load_model()
        self._lock = Lock()
        self.total_rerank_seconds = 0.0
        self.scored_candidate_count = 0

    @staticmethod
    def _weight(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        return normalized

    def _load_model(self) -> _LlamaBackend:
        if not self.model_path.is_file():
            raise LLMRerankerUnavailable(f"LLM model not found: {self.model_path}")
        try:
            from llama_cpp import Llama
        except (ImportError, OSError) as exc:
            raise LLMRerankerUnavailable(f"llama-cpp-python is unavailable: {exc}") from exc
        try:
            return Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_size,
                logits_all=True,
                verbose=False,
            )
        except Exception as exc:
            raise LLMRerankerUnavailable(f"failed to load LLM model {self.model_path}: {exc}") from exc

    @property
    def mean_candidate_latency_ms(self) -> float | None:
        """Return cumulative end-to-end rerank latency per scored candidate."""

        if not self.scored_candidate_count:
            return None
        return self.total_rerank_seconds * 1000.0 / self.scored_candidate_count

    def _tokens(self, text: str, *, add_bos: bool) -> list[int]:
        return list(self._model.tokenize(text.encode("utf-8"), add_bos=add_bos, special=False))

    def _start_token(self) -> int:
        token = self._model.token_bos()
        if token < 0:
            token = self._model.token_eos()
        if token < 0:
            raise ValueError("model has no BOS or EOS token for empty preceding context")
        return token

    def _prepare_tokens(
        self,
        candidates: Sequence[Candidate],
        context: ConversionContext,
    ) -> tuple[list[int], tuple[list[int], ...]]:
        candidate_tokens = tuple(self._tokens(candidate.surface, add_bos=False) for candidate in candidates)
        if any(not tokens for tokens in candidate_tokens):
            raise ValueError("every candidate must tokenize to at least one token")

        max_candidate_tokens = max(map(len, candidate_tokens))
        if max_candidate_tokens >= self.context_size:
            raise ValueError("candidate exceeds the configured LLM context size")
        following_tokens = self._tokens(context.following, add_bos=False)
        following_capacity = self.context_size - max_candidate_tokens - 1
        following_tokens = following_tokens[:following_capacity]
        continuations = tuple(tokens + following_tokens for tokens in candidate_tokens)

        max_continuation_tokens = max(map(len, continuations))
        prefix_capacity = self.context_size - max_continuation_tokens
        prefix_tokens = self._tokens(context.preceding, add_bos=True)
        if not prefix_tokens:
            prefix_tokens = [self._start_token()]
        prefix_tokens = prefix_tokens[-prefix_capacity:]
        return prefix_tokens, continuations

    def _mean_log_likelihood(self, continuation: Sequence[int], prefix_length: int) -> float:
        self._model.eval(continuation)
        total = 0.0
        for offset, token in enumerate(continuation):
            logits = self._model.scores[prefix_length + offset - 1]
            log_probs = self._model.logits_to_logprobs(logits)
            total += float(log_probs[token])
        mean = total / len(continuation)
        if not math.isfinite(mean):
            raise ValueError("model produced a non-finite log-likelihood")
        return mean

    def rerank(
        self,
        candidates: Sequence[Candidate],
        context: ConversionContext,
    ) -> tuple[Candidate, ...]:
        if not isinstance(context, ConversionContext):
            raise TypeError("context must be a ConversionContext")
        materialized = tuple(candidates)
        if not materialized:
            return ()

        with self._lock:
            prefix_tokens, continuations = self._prepare_tokens(materialized, context)
            started = perf_counter()
            self._model.reset()
            self._model.eval(prefix_tokens)
            prefix_state = self._model.save_state()
            scored: list[Candidate] = []
            for candidate, continuation in zip(materialized, continuations, strict=True):
                self._model.load_state(prefix_state)
                mean_log_likelihood = self._mean_log_likelihood(continuation, len(prefix_tokens))
                combined_score = (
                    self.viterbi_weight * candidate.cost
                    - self.llm_weight * mean_log_likelihood
                )
                scored.append(replace(candidate, score=combined_score))
            self.total_rerank_seconds += perf_counter() - started
            self.scored_candidate_count += len(scored)

        return tuple(sorted(scored, key=lambda candidate: (candidate.score, candidate.cost, candidate.surface)))


__all__ = [
    "ConversionContext",
    "DEFAULT_LLM_CONTEXT_SIZE",
    "DEFAULT_LLM_MODEL_PATH",
    "DEFAULT_LLM_WEIGHT",
    "DEFAULT_VITERBI_WEIGHT",
    "LLMLogitsReranker",
    "LLMRerankerUnavailable",
    "MockContextReranker",
    "Reranker",
]
