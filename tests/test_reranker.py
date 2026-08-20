"""Behavioral tests for the Phase 2 reranker contract and mock."""

from __future__ import annotations
import math

from pathlib import Path

import pytest

from beme.converter import Converter
from beme.reranker import (
    ConversionContext,
    LLMLogitsReranker,
    LLMRerankerUnavailable,
    MockContextReranker,
)


def write_dictionary(tmp_path: Path) -> Path:
    path = tmp_path / "dictionary.tsv"
    path.write_text(
        "reading\tsurface\tpos\tcost\n"
        "こうしょう\t交渉\tnoun\t0.0\n"
        "こうしょう\t考証\tnoun\t0.4\n"
        "こうしょう\t校章\tnoun\t0.5\n",
        encoding="utf-8",
    )
    return path


def test_preceding_context_promotes_history_sense_from_full_candidate_pool(tmp_path: Path) -> None:
    converter = Converter.from_tsv(write_dictionary(tmp_path), reranker=MockContextReranker())

    baseline = converter.convert("こうしょう", k=1)
    contextual = converter.convert(
        "こうしょう",
        k=1,
        context=ConversionContext(preceding="歴史資料を考える"),
    )

    assert baseline.candidates[0].surface == "交渉"
    assert baseline.reranked is False
    assert contextual.candidates[0].surface == "考証"
    assert contextual.candidates[0].cost == pytest.approx(0.4)
    assert contextual.candidates[0].score == pytest.approx(-0.6)
    assert contextual.reranked is True


def test_following_context_promotes_business_sense_deterministically(tmp_path: Path) -> None:
    converter = Converter.from_tsv(write_dictionary(tmp_path), reranker=MockContextReranker())
    context = ConversionContext(following="について商談を進める")

    first = converter.convert("こうしょう", k=3, context=context)
    second = converter.convert("こうしょう", k=3, context=context)

    assert first == second
    assert first.candidates[0].surface == "交渉"
    assert first.candidates[0].score == pytest.approx(-1.0)
    assert {candidate.surface for candidate in first.candidates} == {"交渉", "考証", "校章"}


def test_mock_reranker_supports_injected_keyword_lexicon(tmp_path: Path) -> None:
    reranker = MockContextReranker(
        keyword_bonus=2.0,
        keywords={"校章": ("制服",)},
    )
    converter = Converter.from_tsv(write_dictionary(tmp_path), reranker=reranker)

    result = converter.convert(
        "こうしょう",
        k=1,
        context=ConversionContext(preceding="制服を確認する"),
    )

    assert result.candidates[0].surface == "校章"
    assert result.candidates[0].score == pytest.approx(-1.5)


def test_context_rejects_non_string_fields() -> None:
    with pytest.raises(TypeError, match="preceding context"):
        ConversionContext(preceding=1)  # type: ignore[arg-type]



class FakeLlama:
    """Minimal deterministic backend; no model file or llama runtime used."""

    def __init__(self) -> None:
        self.n_tokens = 0
        self.scores = [[0.0] * 4 for _ in range(16)]
        self.save_calls = 0
        self.load_calls = 0
        self.evaluated: list[list[int]] = []

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        del special
        if add_bos:
            return [0]
        return {
            "交渉".encode(): [1],
            "考証".encode(): [2],
            "校章".encode(): [1, 1],
            "の記録".encode(): [3],
            b"": [],
        }[text]

    def token_bos(self) -> int:
        return 0

    def token_eos(self) -> int:
        return 0

    def reset(self) -> None:
        self.n_tokens = 0

    def eval(self, tokens: list[int] | tuple[int, ...]) -> None:
        self.evaluated.append(list(tokens))
        logits = [-4.0, -1.5, 0.0, -4.0]
        for token in tokens:
            self.scores[self.n_tokens] = logits.copy()
            self.n_tokens += 1

    def save_state(self) -> tuple[int, list[list[float]]]:
        self.save_calls += 1
        return self.n_tokens, [row.copy() for row in self.scores]

    def load_state(self, state: tuple[int, list[list[float]]]) -> None:
        self.load_calls += 1
        self.n_tokens, scores = state
        self.scores = [row.copy() for row in scores]

    def logits_to_logprobs(self, logits: list[float], axis: int = -1) -> list[float]:
        del axis
        maximum = max(logits)
        normalization = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
        return [value - normalization for value in logits]


def test_llm_logits_reranker_reuses_prefilled_state_and_combines_scores(tmp_path: Path) -> None:
    model = FakeLlama()
    reranker = LLMLogitsReranker(
        model= model,
        context_size=16,
        llm_weight=1.0,
        viterbi_weight=1.0,
    )
    converter = Converter.from_tsv(write_dictionary(tmp_path), reranker=reranker)

    result = converter.convert(
        "こうしょう",
        k=3,
        context=ConversionContext(preceding="前文", following="の記録"),
    )

    assert result.candidates[0].surface == "考証"
    assert model.save_calls == 1
    assert model.load_calls == 3
    assert [1, 3] in model.evaluated
    assert [2, 3] in model.evaluated
    assert reranker.scored_candidate_count == 3
    assert reranker.mean_candidate_latency_ms is not None


def test_llm_logits_reranker_reports_missing_model_without_importing_runtime(tmp_path: Path) -> None:
    with pytest.raises(LLMRerankerUnavailable, match="model not found"):
        LLMLogitsReranker(tmp_path / "missing.gguf")


@pytest.mark.parametrize(
    ("keyword", "value"),
    (("llm_weight", -1.0), ("viterbi_weight", float("inf"))),
)
def test_llm_logits_reranker_rejects_invalid_weights(keyword: str, value: float) -> None:
    arguments = {keyword: value}
    with pytest.raises(ValueError):
        LLMLogitsReranker(model=FakeLlama(), **arguments)