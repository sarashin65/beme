from scripts.e0_llm_prompt import RULES_VERSION, SYSTEM_PROMPT, build_llm_messages


def test_build_prompt_contains_contract_and_case_data() -> None:
    case = {"id": "case-1", "source_text": "研究を行った。"}
    result = build_llm_messages(case, [{"surface": "研究", "reading": "けんきゅう"}], [{"id": "a1"}])
    assert result["payload"]["case_id"] == "case-1"
    assert result["payload"]["source_sentence"] == "研究を行った。"
    assert result["payload"]["rules_version"] == RULES_VERSION
    assert "一文字も変更" in SYSTEM_PROMPT
    assert "JSON only" in SYSTEM_PROMPT
    assert "few_shot_examples" in result["user"]
    assert "研究を行った。" in result["user"]


def test_prompt_escapes_special_characters_as_json() -> None:
    sentence = '「A\\B」\n次？'
    result = build_llm_messages({"case_id": "特殊", "source_sentence": sentence}, [], [])
    assert sentence in result["payload"]["source_sentence"]
    assert "\\n" in result["user"]
    assert "特殊" in result["user"]
