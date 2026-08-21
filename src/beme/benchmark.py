"""Sentence-level AJIMEE-Bench and Mozc regression evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .dictionary import is_hiragana
from .reranker import ConversionContext


_AJIMEE_REQUIRED_KEYS = frozenset(
    {
        "index",
        "context_text",
        "input",
        "expected_output",
        "original_text",
        "splitted_input_for_limited_input_length",
    }
)
_CONVERSION_EXPECTED = re.compile(r"^Conversion Expected(?: ([1-9]\d*))?$")
_SUPPORTED_MOZC_COMMANDS = frozenset(
    {"Conversion Match", "Conversion Not Match", "Conversion Not Expected"}
)


@dataclass(frozen=True, slots=True)
class AjimeeCase:
    """One AJIMEE-Bench JWTD v2 evaluation item."""

    id: str
    context: str
    input: str
    references: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MozcCase:
    """One Mozc quality-regression item."""

    id: str
    label: str
    reading: str
    expected: str
    command: str
    expected_rank: int | None

    @property
    def supported(self) -> bool:
        return self.expected_rank is not None or self.command in _SUPPORTED_MOZC_COMMANDS


@dataclass(frozen=True, slots=True)
class SentenceCandidate:
    """A composed sentence candidate from one or more convertible spans."""

    surface: str
    cost: float


def katakana_to_hiragana(text: str) -> str:
    """Convert standard full-width katakana to hiragana, preserving literals."""

    converted: list[str] = []
    for char in text:
        codepoint = ord(char)
        if 0x30A1 <= codepoint <= 0x30F6:
            converted.append(chr(codepoint - 0x60))
        elif char == "ヽ":
            converted.append("ゝ")
        elif char == "ヾ":
            converted.append("ゞ")
        else:
            converted.append(char)
    return "".join(converted)


def _input_spans(text: str) -> Iterable[tuple[bool, str]]:
    start = 0
    convertible = is_hiragana(text[0])
    for index in range(1, len(text)):
        current = is_hiragana(text[index])
        if current != convertible:
            yield convertible, text[start:index]
            start = index
            convertible = current
    yield convertible, text[start:]


def _deduplicate_candidates(candidates: Iterable[SentenceCandidate], k: int) -> tuple[SentenceCandidate, ...]:
    by_surface: dict[str, SentenceCandidate] = {}
    for candidate in sorted(candidates, key=lambda item: (item.cost, item.surface)):
        by_surface.setdefault(candidate.surface, candidate)
        if len(by_surface) == k:
            break
    return tuple(by_surface.values())


def convert_sentence(
    converter: Any,
    input_text: str,
    *,
    k: int = 5,
    cascade_threshold: float = 2.0,
    preceding_context: str = "",
) -> tuple[SentenceCandidate, ...]:
    """Convert kana spans and pass punctuation, digits, and symbols through.

    AJIMEE uses katakana readings while BeME accepts hiragana, so katakana is
    normalized before conversion.  Non-kana literals delimit independent
    lattice spans.  Their K-best outputs are composed and pruned globally.
    """

    if not isinstance(input_text, str) or not input_text:
        raise ValueError("input_text must be a non-empty string")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not isinstance(preceding_context, str):
        raise TypeError("preceding_context must be a string")

    normalized = katakana_to_hiragana(input_text)
    candidates = (SentenceCandidate("", 0.0),)
    context = ConversionContext(preceding=preceding_context)
    for convertible, span in _input_spans(normalized):
        if not convertible:
            candidates = tuple(
                SentenceCandidate(candidate.surface + span, candidate.cost)
                for candidate in candidates
            )
            continue
        result = converter.convert(
            span,
            k=k,
            cascade_threshold=cascade_threshold,
            context=context,
        )
        combined = (
            SentenceCandidate(prefix.surface + suffix.surface, prefix.cost + suffix.cost)
            for prefix in candidates
            for suffix in result.candidates
        )
        candidates = _deduplicate_candidates(combined, k)
    return candidates


def edit_distance(reference: str, hypothesis: str) -> int:
    """Return Levenshtein distance using linear auxiliary memory."""

    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must be strings")
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_char in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + (reference_char != hypothesis_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Return CER = Levenshtein distance / reference character count."""

    if not reference:
        return 0.0 if not hypothesis else float("inf")
    return edit_distance(reference, hypothesis) / len(reference)


def load_ajimee_cases(path: str | Path) -> list[AjimeeCase]:
    """Load and validate AJIMEE-Bench JWTD v2 JSON."""

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("AJIMEE root must be an array")

    cases: list[AjimeeCase] = []
    seen_ids: set[str] = set()
    for position, item in enumerate(payload, start=1):
        if not isinstance(item, dict) or not _AJIMEE_REQUIRED_KEYS.issubset(item):
            raise ValueError(f"AJIMEE item {position}: missing required fields")
        index = item["index"]
        context = item["context_text"]
        input_text = item["input"]
        references = item["expected_output"]
        if not isinstance(index, str) or not index:
            raise ValueError(f"AJIMEE item {position}: index must be a non-empty string")
        case_id = f"ajimee-jwtd-v2-{index}"
        if case_id in seen_ids:
            raise ValueError(f"AJIMEE item {position}: duplicate index {index!r}")
        seen_ids.add(case_id)
        if not isinstance(context, str):
            raise ValueError(f"AJIMEE item {position}: context_text must be a string")
        if not isinstance(input_text, str) or not input_text:
            raise ValueError(f"AJIMEE item {position}: input must be a non-empty string")
        if (
            not isinstance(references, list)
            or not references
            or any(not isinstance(reference, str) or not reference for reference in references)
        ):
            raise ValueError(f"AJIMEE item {position}: expected_output must contain strings")
        cases.append(AjimeeCase(case_id, context, input_text, tuple(dict.fromkeys(references))))
    return cases


def evaluate_ajimee(
    converter: Any,
    cases: Iterable[AjimeeCase],
    *,
    k: int = 5,
    cascade_threshold: float = 2.0,
) -> dict[str, Any]:
    """Evaluate AJIMEE items with sentence-level exact match, distance, and CER."""

    per_case: list[dict[str, Any]] = []
    total_cer = 0.0
    total_distance = 0
    total_exact = 0
    for case in cases:
        candidates = convert_sentence(
            converter,
            case.input,
            k=k,
            cascade_threshold=cascade_threshold,
            preceding_context=case.context,
        )
        hypothesis = candidates[0].surface
        distances = [edit_distance(reference, hypothesis) for reference in case.references]
        cers = [distance / len(reference) for distance, reference in zip(distances, case.references)]
        minimum_distance = min(distances)
        minimum_cer = min(cers)
        exact = hypothesis in case.references
        total_distance += minimum_distance
        total_cer += minimum_cer
        total_exact += int(exact)
        per_case.append(
            {
                "id": case.id,
                "context": case.context,
                "input": case.input,
                "normalized_input": katakana_to_hiragana(case.input),
                "references": list(case.references),
                "hypothesis": hypothesis,
                "edit_distance": minimum_distance,
                "cer": minimum_cer,
                "exact_match": exact,
            }
        )

    case_count = len(per_case)
    denominator = case_count or 1
    return {
        "suite": "ajimee",
        "case_count": case_count,
        "k": k,
        "mean_cer": total_cer / denominator,
        "mean_edit_distance": total_distance / denominator,
        "exact_match_accuracy": total_exact / denominator,
        "per_case": per_case,
    }


def load_mozc_cases(path: str | Path) -> list[MozcCase]:
    """Load Mozc oss.tsv and retain unsupported commands as explicit skips."""

    cases: list[MozcCase] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if not 4 <= len(fields) <= 7 or any(not field for field in fields[:4]):
                raise ValueError(f"Mozc line {line_number}: expected four to seven TSV fields")
            label, reading, expected, command = fields[:4]
            expected_match = _CONVERSION_EXPECTED.fullmatch(command)
            expected_rank = None
            if expected_match:
                rank_field = fields[4] if len(fields) >= 5 else expected_match.group(1)
                try:
                    expected_rank = int(rank_field or 0)
                except ValueError as exc:
                    raise ValueError(f"Mozc line {line_number}: expected rank must be an integer") from exc
            cases.append(
                MozcCase(
                    id=f"mozc-{line_number}",
                    label=label,
                    reading=reading,
                    expected=expected,
                    command=command,
                    expected_rank=expected_rank,
                )
            )
    return cases


def _mozc_case_passed(case: MozcCase, surfaces: list[str]) -> bool:
    top = surfaces[0]
    if case.command == "Conversion Match":
        return case.expected in top
    if case.command == "Conversion Not Match":
        return case.expected not in top
    if case.command == "Conversion Not Expected":
        return case.expected not in surfaces
    if case.expected_rank is not None:
        return case.expected in surfaces[: case.expected_rank + 1]
    raise ValueError(f"unsupported Mozc command {case.command!r}")


def evaluate_mozc(
    converter: Any,
    cases: Iterable[MozcCase],
    *,
    k: int = 5,
    cascade_threshold: float = 2.0,
) -> dict[str, Any]:
    """Run supported Mozc conversion regressions and report explicit skips."""

    per_case: list[dict[str, Any]] = []
    passed_count = 0
    evaluated_count = 0
    skipped_commands: dict[str, int] = {}
    for case in cases:
        if not case.supported:
            skipped_commands[case.command] = skipped_commands.get(case.command, 0) + 1
            per_case.append(
                {
                    "id": case.id,
                    "label": case.label,
                    "reading": case.reading,
                    "expected": case.expected,
                    "command": case.command,
                    "status": "skipped",
                    "reason": "BeME has no suggestion/prediction mode",
                }
            )
            continue
        candidates = convert_sentence(
            converter,
            case.reading,
            k=k,
            cascade_threshold=cascade_threshold,
        )
        surfaces = [candidate.surface for candidate in candidates]
        passed = _mozc_case_passed(case, surfaces)
        evaluated_count += 1
        passed_count += int(passed)
        per_case.append(
            {
                "id": case.id,
                "label": case.label,
                "reading": case.reading,
                "expected": case.expected,
                "command": case.command,
                "expected_max_rank": case.expected_rank,
                "retrieved": surfaces,
                "passed": passed,
                "status": "passed" if passed else "failed",
            }
        )

    denominator = evaluated_count or 1
    return {
        "suite": "mozc",
        "case_count": len(per_case),
        "evaluated_count": evaluated_count,
        "passed_count": passed_count,
        "pass_rate": passed_count / denominator,
        "skipped_count": len(per_case) - evaluated_count,
        "skipped_commands": skipped_commands,
        "k": k,
        "per_case": per_case,
    }


__all__ = [
    "AjimeeCase",
    "MozcCase",
    "SentenceCandidate",
    "character_error_rate",
    "convert_sentence",
    "edit_distance",
    "evaluate_ajimee",
    "evaluate_mozc",
    "katakana_to_hiragana",
    "load_ajimee_cases",
    "load_mozc_cases",
]
