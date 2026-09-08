"""Prompt and output contract for E0 LLM annotation."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "e0-annotation-v1"
RULES_VERSION = "e0-boundary-v1"

SYSTEM_PROMPT = f"""You are the E0 Japanese reading and segmentation annotator.
Return JSON only. Do not output Markdown, explanations, or chain-of-thought.

Rules:
1. Preserve source_sentence exactly, character for character (原文を一文字も変更しない). Never edit, normalize,
   correct, paraphrase, or omit the original surface text.
2. Give the natural contextual reading in hiragana. Keep punctuation, symbols, and
   digits according to the boundary specification and mark them literal when they
   are not lexical dictionary tokens.
3. Tokenize with IPADIC-compatible morphological boundaries: separate particles and
   auxiliary verbs; keep a dictionary compound as one token when that is the
   compatible morphological unit; use the supplied MeCab result only as a fallible
   proposal, not as the answer.
4. If a proper noun, unusual spelling, ateji, technical term, or other reading cannot
   be established from context, do not guess. Set needs_review to true and explain the
   uncertainty briefly in review_reasons; still report only evidence-supported tokens.
5. Offsets are Unicode code-point offsets, half-open [start, end), and every token
   must be contiguous in both surface and reading. canonical_reference must be
   byte-for-byte identical to source_sentence.

Output schema version: {SCHEMA_VERSION}. The required top-level fields are
schema_version, case_id, input_reading, canonical_reference, and tokens. Each token
has surface, reading, surface_start, surface_end, reading_start, reading_end, and
is_literal. Optional review fields are allowed only as specified by the user payload.
Do not include any fields outside the schema.

Fixed boundary specification version: {RULES_VERSION}.
"""

FEW_SHOT = [
    {
        "label": "compound noun",
        "source_sentence": "情報処理を行う。",
        "output": {"input_reading": "じょうほうしょりをおこなう。", "tokens": [
            {"surface": "情報処理", "reading": "じょうほうしょり", "is_literal": False},
            {"surface": "を", "reading": "を", "is_literal": False},
            {"surface": "行う", "reading": "おこなう", "is_literal": False},
            {"surface": "。", "reading": "。", "is_literal": True},
        ]},
    },
    {
        "label": "inflection and counter",
        "source_sentence": "三冊読みました。",
        "output": {"input_reading": "さんさつよみました。", "tokens": [
            {"surface": "三", "reading": "三", "is_literal": True},
            {"surface": "冊", "reading": "さつ", "is_literal": False},
            {"surface": "読み", "reading": "よみ", "is_literal": False},
            {"surface": "まし", "reading": "まし", "is_literal": False},
            {"surface": "た", "reading": "た", "is_literal": False},
            {"surface": "。", "reading": "。", "is_literal": True},
        ]},
    },
    {
        "label": "digits symbols katakana",
        "source_sentence": "AIは2026年版！",
        "output": {"input_reading": "AIは2026年版！", "tokens": [
            {"surface": "AI", "reading": "AI", "is_literal": True},
            {"surface": "は", "reading": "は", "is_literal": False},
            {"surface": "2026", "reading": "2026", "is_literal": True},
            {"surface": "年", "reading": "ねん", "is_literal": False},
            {"surface": "版", "reading": "ばん", "is_literal": False},
            {"surface": "！", "reading": "！", "is_literal": True},
        ]},
    },
    {
        "label": "proper noun and ateji",
        "source_sentence": "東京と五月雨。",
        "output": {"input_reading": "とうきょうとさみだれ。", "tokens": [
            {"surface": "東京", "reading": "とうきょう", "is_literal": False},
            {"surface": "と", "reading": "と", "is_literal": False},
            {"surface": "五月雨", "reading": "さみだれ", "is_literal": False},
            {"surface": "。", "reading": "。", "is_literal": True},
        ]},
    },
]


def _payload(case: Mapping[str, Any], mecab_tokens: Sequence[Mapping[str, Any]], d1_anchors: Sequence[Any]) -> dict[str, Any]:
    return {
        "case_id": str(case.get("case_id", case.get("id", ""))),
        "source_sentence": str(case.get("source_sentence", case.get("source_text", case.get("canonical_reference", "")))),
        "mecab_tokens": list(mecab_tokens),
        "d1_anchors": list(d1_anchors),
        "rules_version": RULES_VERSION,
    }


def build_llm_messages(case: Mapping[str, Any], mecab_tokens: Sequence[Mapping[str, Any]], d1_anchors: Sequence[Any]) -> dict[str, Any]:
    """Build deterministic system/user prompts and their structured payload."""
    payload = _payload(case, mecab_tokens, d1_anchors)
    user = (
        "Annotate the following case under the fixed rules. The examples are format "
        "guidance only; do not copy their content.\n\n"
        f"few_shot_examples={json.dumps(FEW_SHOT, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"case={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
    return {
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user}],
        "user": user,
        "payload": payload,
    }


__all__ = ["SCHEMA_VERSION", "RULES_VERSION", "SYSTEM_PROMPT", "build_llm_messages"]
