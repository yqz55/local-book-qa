"""Two-sentence anaphora safety and exact source-range contracts."""

from copy import deepcopy

import pytest

from evidence.joint_extraction import JOINT_POLICY, JOINT_RULES, extract_joint
from evidence.dag_tokenization import DAGTokenizer
from evidence.ingest import SourceMap, clean_text
from evidence.questions import analyze_question
from tests.test_answer_rules import corpus


def joint(source, question="青禾社区雨水收集项目什么时候启动？", **kwargs):
    index = corpus(source, **kwargs)
    seg = index["segmentation"]
    tokenizer = DAGTokenizer(seg["lexicon"], index["dictionary"]["book_words"], [],
                             index["dictionary"]["book_words"], seg["seed_collection_tf"],
                             book_bonus_cap=seg["config"]["book_bonus_cap"])
    cleaned = clean_text(question)
    tokens, _ = tokenizer.tokenize(cleaned["text"], mapping=SourceMap(cleaned["source_runs"], question), capture=True)
    result = {"question": analyze_question(cleaned, question, tokens), "query_cleaning": cleaned, "query_ambiguities": []}
    # Isolate joint rules from ranking: both adjacent sentences are explicitly
    # eligible, including neighbors with no literal query match of their own.
    evidence = {"candidates": [{**s, "sentence_id": s["id"]} for s in index["sentences"]["sentences"]]}
    return index, result, evidence, extract_joint(index, result, evidence)


def accepted(decisions):
    return [decision for decision in decisions if decision["accepted"]]


def assert_spans(index, decision):
    sentences = {s["id"]: s for s in index["sentences"]["sentences"]}
    for field in decision["fields"].values():
        assert index["source"]["text"][field["source_start"]:field["source_end"]] == field["text"]
        assert index["cleaning"]["text"][field["clean_start"]:field["clean_end"]]
        sentence = sentences[field["sentence_id"]]
        assert sentence["clean_start"] <= field["clean_start"] < field["clean_end"] <= sentence["clean_end"]
        assert sentence["source_start"] <= field["source_start"] < field["source_end"] <= sentence["source_end"]


@pytest.mark.parametrize("separator", ["", "\n", "\n\n", "\r\n"])
def test_named_subject_and_date_can_cross_ordinary_physical_lines(separator):
    source = "青禾社区雨水收集项目进入实施阶段。" + separator + "该项目于2025年3月12日启动。"
    index, _, _, decisions = joint(source)
    [decision] = accepted(decisions)
    assert decision["rule"] == "J101"
    assert decision["fields"]["time"]["text"] == "2025年3月12日"
    assert decision["fields"]["subject"]["text"] == "青禾社区雨水收集项目"
    assert decision["fields"]["reference"]["text"] == "该项目"
    assert decision["fields"]["time"]["sentence_id"] != decision["fields"]["subject"]["sentence_id"]
    assert_spans(index, decision)


@pytest.mark.parametrize("name,reference", [("青禾试点", "该试点"), ("青禾活动", "该活动"), ("青禾实验", "该实验"), ("青禾工程", "该工程")])
def test_supported_object_types_have_explicit_compatible_references(name, reference):
    index, _, _, decisions = joint(f"{name}进入实施阶段。\n{reference}于2025年启动。", f"{name}何时启动？")
    [decision] = accepted(decisions)
    assert decision["fields"]["time"]["text"] == "2025年"
    assert_spans(index, decision)


@pytest.mark.parametrize("value,unit", [("300元", "元"), ("约300万元", "万元"), ("10至12元", "元"), ("3.14元", "元"), ("不超过300元", "元"), ("300元以上", "元")])
def test_quantity_keeps_original_number_range_comparison_and_unit(value, unit):
    index, _, _, decisions = joint(f"青禾项目进入实施阶段。\n该项目预算为{value}。", f"青禾项目预算是多少{unit}？")
    [decision] = accepted(decisions)
    assert decision["rule"] == "J102"
    assert decision["fields"]["value"]["text"] == value
    assert_spans(index, decision)


def test_explicit_unit_mismatch_is_rejected():
    _, _, _, decisions = joint("青禾项目进入实施阶段。\n该项目预算为300升。", "青禾项目预算是多少元？")
    assert not accepted(decisions)
    assert any("单位" in d["reason"] for d in decisions)


@pytest.mark.parametrize("left", [
    "青禾项目和松林项目进入实施阶段。",
    "青禾项目与松林项目进入实施阶段。",
    "青禾项目进入实施阶段，松林工程随后开始。",
    "青禾项目组进入实施阶段。",
    "青禾项目的负责人王明完成工作。",
    "青禾项目启动日期是2025年。",
])
def test_multiple_or_non_subject_antecedents_are_rejected(left):
    _, _, _, decisions = joint(left + "该项目于2025年启动。", "青禾项目何时启动？")
    assert not accepted(decisions)
    assert decisions


@pytest.mark.parametrize("second", ["该试点于2025年启动。", "它于2025年启动。", "他于2025年启动。", "她于2025年启动。", "随后，该项目于2025年启动。"])
def test_no_general_pronouns_or_reference_type_coercion(second):
    _, _, _, decisions = joint("青禾项目进入实施阶段。" + second, "青禾项目何时启动？")
    assert not accepted(decisions)


def test_both_adjacent_sentences_must_be_eligible():
    index, result, evidence, _ = joint("青禾项目进入实施阶段。该项目于2025年启动。", "青禾项目何时启动？")
    for remaining in (evidence["candidates"][:1], evidence["candidates"][1:]):
        assert not extract_joint(index, result, {"candidates": remaining})


def test_nonadjacent_sentences_never_bind_even_with_both_in_scope():
    _, _, _, decisions = joint("青禾项目进入实施阶段。天气晴朗。该项目于2025年启动。", "青禾项目何时启动？")
    assert not accepted(decisions)


def test_frozen_context_boundary_prevents_joint_extraction():
    index, result, evidence, _ = joint("青禾项目进入实施阶段。该项目于2025年启动。", "青禾项目何时启动？")
    left, right = index["sentences"]["sentences"]
    index["context"] = {"boundaries": [{"left_sentence_id": left["id"], "right_sentence_id": right["id"], "reasons": ["CW004"]}]}
    decisions = extract_joint(index, result, evidence)
    assert not accepted(decisions)
    assert decisions[0]["binding"]["checks"]["boundary_reasons"] == ["CW004"]


@pytest.mark.parametrize("separator", ["\n第二章 新项目\n", "\n1. 该项目于2025年启动。\n"])
def test_chapter_or_list_text_never_bridges_subject(separator):
    _, _, _, decisions = joint("青禾项目进入实施阶段。" + separator + "该项目于2026年启动。", "青禾项目何时启动？")
    assert not accepted(decisions)


@pytest.mark.parametrize("second", [
    "该项目未于2025年启动。",
    "该项目计划于2025年启动。",
    "该项目将于2025年启动。",
    "该项目可能于2025年启动。",
    "该项目据说于2025年启动。",
    "该项目于2025年启动？",
    "该项目于2025年启动",
])
def test_nonfactual_or_fragmentary_matching_pairs_expose_rejection(second):
    _, _, _, decisions = joint("青禾项目进入实施阶段。" + second, "青禾项目何时启动？")
    assert not accepted(decisions)
    # Context windows block fragment pairs before the factual guard.  Complete
    # negative/planned/question clauses retain the factual rejection itself.
    if second.endswith(("。", "？")):
        assert decisions[0]["rule"] == "JG101"
        assert decisions[0]["binding"]["checks"]["unsafe_reasons"]
    else:
        assert decisions[0]["rule"] in {"JG101", "JG102"}


def test_nonfactual_antecedent_also_prevents_fact_answer():
    _, _, _, decisions = joint("青禾项目计划进入实施阶段。该项目于2025年启动。", "青禾项目何时启动？")
    assert not accepted(decisions)
    assert decisions[0]["rule"] == "JG101"


def test_direct_contradiction_remains_a_matching_unsafe_pair_for_final_guard():
    _, _, _, decisions = joint("青禾项目进入实施阶段。该项目于2025年启动。\n青禾项目进入实施阶段。该项目并未于2025年启动。", "青禾项目何时启动？")
    assert len(accepted(decisions)) == 1
    assert any(d["rule"] == "JG101" and d["binding"]["checks"]["matching_requested_predicate"] for d in decisions)


def test_explicit_named_factual_date_can_anchor_next_reference_without_hiding_conflict():
    _, _, _, decisions = joint("青禾项目于2025年启动。该项目于2026年启动。", "青禾项目何时启动？")
    [decision] = accepted(decisions)
    assert decision["fields"]["time"]["text"] == "2026年"


def test_event_before_date_is_an_explicit_supported_time_shape():
    index, _, _, decisions = joint("青禾项目进入实施阶段。该项目启动于2025年。", "青禾项目何时启动？")
    [decision] = accepted(decisions)
    assert decision["fields"]["time"]["text"] == "2025年"
    assert_spans(index, decision)


@pytest.mark.parametrize("second", [
    "该项目于2025年和2026年启动。",
    "该项目启动于2025年，另一活动于2026年举行。",
])
def test_multiple_or_unrelated_dates_do_not_select_first(second):
    _, _, _, decisions = joint("青禾项目进入实施阶段。" + second, "青禾项目何时启动？")
    assert not accepted(decisions)


def test_different_event_does_not_become_queried_event_date():
    _, _, _, decisions = joint("青禾项目进入实施阶段。该项目于2025年结束。", "青禾项目何时启动？")
    assert not accepted(decisions)
    assert decisions[0]["rule"] == "JG102"


def test_source_ranges_remain_real_after_cleaning_and_retrieval_block_split():
    source = "\ufeff　青禾社区雨水收集项目进入实施阶段。　\r\n　该项目于2025年3月12日启动。"
    index, _, _, decisions = joint(source, max_chars=16)
    [decision] = accepted(decisions)
    assert_spans(index, decision)
    assert decision["binding"]["antecedent"] == decision["fields"]["subject"]
    assert decision["binding"]["reference"] == decision["fields"]["reference"]


def test_subject_cannot_match_inside_longer_protected_source_word():
    _, _, _, decisions = joint("青禾项目启动进入实施阶段。该项目于2025年启动。", "青禾项目何时启动？", book_words=["青禾项目启动"])
    assert not accepted(decisions)
    assert not decisions[0]["binding"]["checks"]["actual_term_boundaries"]


@pytest.mark.parametrize("modifier,source_modifier", [("正式", ""), ("成功", ""), ("正式", "成功"), ("成功", "正式")])
def test_time_question_content_modifier_is_never_silently_dropped(modifier, source_modifier):
    _, _, _, decisions = joint(f"青禾项目进入实施阶段。该项目于2025年{source_modifier}启动。", f"青禾项目什么时候{modifier}启动？")
    assert not accepted(decisions)
    assert decisions[0]["binding"]["checks"]["required_modifier"] == modifier
    assert not decisions[0]["binding"]["checks"]["matching_event_modifier"]


@pytest.mark.parametrize("modifier", ["正式", "成功"])
def test_time_question_modifier_retained_with_matching_source_predicate(modifier):
    index, _, _, decisions = joint(f"青禾项目进入实施阶段。该项目于2025年{modifier}启动。", f"青禾项目什么时候{modifier}启动？")
    [decision] = accepted(decisions)
    assert decision["binding"]["checks"]["matching_event_modifier"]
    assert decision["fields"]["time"]["text"] == "2025年"
    assert_spans(index, decision)


def test_generic_time_question_allows_explicit_source_modifier():
    _, _, _, decisions = joint("青禾项目进入实施阶段。该项目于2025年正式启动。", "青禾项目何时启动？")
    [decision] = accepted(decisions)
    assert "required_modifier" not in decision["binding"]["checks"]


def test_query_constraints_and_input_snapshot_are_preserved():
    index, result, evidence, _ = joint("青禾项目进入实施阶段。该项目于2025年启动。", "青禾项目何时启动？")
    before = deepcopy((index, result, evidence))
    result["question"]["constraints"] = [{"kind": "negative", "text": "没有"}]
    changed = deepcopy(result)
    assert not accepted(extract_joint(index, result, evidence))
    assert index == before[0] and result == changed and evidence == before[2]
    assert JOINT_POLICY and set(JOINT_RULES) == {"J101", "J102", "JG101", "JG102", "JG103"}
