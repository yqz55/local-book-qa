"""Causal direction, grammatical roles, source spans and procedure modality."""

from copy import deepcopy

import pytest

from evidence.discourse import (DISCOURSE_CONFIG, DISCOURSE_PATTERNS,
                               DISCOURSE_POLICY, DISCOURSE_RULES, extract_discourse)
from evidence.dag_tokenization import DAGTokenizer
from evidence.ingest import SourceMap, clean_text
from evidence.questions import analyze_question
from tests.test_answer_rules import corpus


def discourse(source, question, **kwargs):
    index = corpus(source, **kwargs)
    seg = index["segmentation"]
    tokenizer = DAGTokenizer(seg["lexicon"], index["dictionary"]["book_words"], [],
                             index["dictionary"]["book_words"], seg["seed_collection_tf"],
                             book_bonus_cap=seg["config"]["book_bonus_cap"])
    cleaned = clean_text(question)
    tokens, _ = tokenizer.tokenize(cleaned["text"], mapping=SourceMap(cleaned["source_runs"], question), capture=True)
    result = {"question": analyze_question(cleaned, question, tokens),
              "query_cleaning": cleaned, "query_ambiguities": []}
    evidence = {"candidates": [{**s, "sentence_id": s["id"]} for s in index["sentences"]["sentences"]]}
    return index, result, evidence, extract_discourse(index, result, evidence)


def accepted(decisions):
    return [d for d in decisions if d["accepted"]]


def exact(index, decision):
    sentences = {s["id"]: s for s in index["sentences"]["sentences"]}
    for field in decision["fields"].values():
        assert index["source"]["text"][field["source_start"]:field["source_end"]] == field["text"]
        assert index["cleaning"]["text"][field["clean_start"]:field["clean_end"]]
        sentence = sentences[field["sentence_id"]]
        assert sentence["clean_start"] <= field["clean_start"] < field["clean_end"] <= sentence["clean_end"]
        assert sentence["source_start"] <= field["source_start"] < field["source_end"] <= sentence["source_end"]


@pytest.mark.parametrize("source,cause,rule", [
    ("因为施工道路积水，所以青禾项目延期启动。", "施工道路积水", "D101"),
    ("由于施工道路积水，青禾项目延期启动。", "施工道路积水", "D101"),
    ("青禾项目延期启动是因为施工道路积水。", "施工道路积水", "D102"),
    ("青禾项目延期启动的原因是施工道路积水。", "施工道路积水", "D102"),
    ("青禾项目延期启动，因为施工道路积水。", "施工道路积水", "D102"),
    ("施工道路积水导致青禾项目延期启动。", "施工道路积水", "D103"),
])
def test_explicit_cause_patterns_preserve_direction_and_fields(source, cause, rule):
    index, _, _, decisions = discourse(source, "青禾项目为什么会延期启动？")
    [decision] = accepted(decisions)
    assert decision["rule"] == rule
    assert decision["fields"]["reason"]["text"] == cause
    assert decision["fields"]["effect"]["text"] == "青禾项目延期启动"
    assert decision["statement_kind"] == "fact"
    exact(index, decision)


@pytest.mark.parametrize("cause", ["备用电源未及时接通", "没有降雨", "降雨不足和灌溉中断"])
def test_negation_and_explicit_joint_cause_are_retained_inside_cause(cause):
    index, _, _, decisions = discourse(f"因为{cause}，所以河道干涸。", "河道为什么干涸？")
    [decision] = accepted(decisions)
    assert decision["fields"]["reason"]["text"] == cause
    exact(index, decision)


@pytest.mark.parametrize("effect,question", [
    ("青禾展馆暂停开放", "青禾展馆为何暂停开放？"),
    ("预约活动取消", "预约活动为什么取消？"),
    ("列车停运", "列车为何停运？"),
])
def test_asked_cancellation_is_itself_an_explicit_event(effect, question):
    _, _, _, decisions = discourse(f"设备损坏导致{effect}。", question)
    assert len(accepted(decisions)) == 1


@pytest.mark.parametrize("source,question", [
    ("没有降雨导致河道干涸。", "没有降雨为什么发生？"),
    ("青禾项目延期启动导致道路积水。", "青禾项目为何延期启动？"),
    ("为了节约用水，居民收集雨水。", "居民为什么收集雨水？"),
    ("降雨量与河道干涸相关。", "河道为什么干涸？"),
    ("因为降雨不足，所以河道没有干涸。", "河道为什么干涸？"),
    ("因为道路积水，所以青禾项目取消启动。", "青禾项目为何启动？"),
    ("因为林青受伤，所以周宁帮助林青。", "林青为何帮助周宁？"),
    ("因为林青受伤，所以林青的朋友帮助周宁。", "林青为何帮助周宁？"),
    ("因为道路积水，所以青禾项目启动。", "青禾项目为何正式启动？"),
])
def test_direction_purpose_negation_roles_and_dropped_qualifiers_never_prove_cause(source, question):
    _, _, _, decisions = discourse(source, question)
    assert not accepted(decisions)


@pytest.mark.parametrize("source", [
    "因为入口受损或电源中断，所以青禾展馆暂停开放。",
    "因为入口可能受损，所以青禾展馆暂停开放。",
    "因为入口受损，所以青禾展馆将暂停开放。",
    "如果入口受损，那么青禾展馆暂停开放。",
    "青禾展馆暂停开放并不是因为入口受损。",
    "据说因为入口受损，所以青禾展馆暂停开放。",
    "“因为入口受损，所以青禾展馆暂停开放。”",
])
def test_nonfactual_alternative_and_negated_causes_stop_fact_answer(source):
    _, _, _, decisions = discourse(source, "青禾展馆为什么暂停开放？")
    assert not accepted(decisions)


@pytest.mark.parametrize("prefix,connector", [
    ("", "未导致"), ("", "未曾导致"), ("", "从未导致"), ("", "不曾导致"),
    ("", "未能直接导致"), ("", "并不必然导致"), ("", "未曾造成"),
    ("并非", "导致"), ("并不是", "导致"),
])
def test_negated_causal_connector_and_assertion_scope_cannot_become_a_cause(prefix, connector):
    _, _, _, decisions = discourse(prefix + "入口处的水管破裂" + connector + "青禾展馆暂停开放。", "青禾展馆为何暂停开放？")
    assert not accepted(decisions)
    assert any(d["blocking"] == "unsafe" for d in decisions)


@pytest.mark.parametrize("marker", ["通过", "采用", "利用", "使用"])
def test_common_method_markers_bind_subject_and_goal(marker):
    index, _, _, decisions = discourse(f"青禾小组{marker}分区采样完成土壤调查。", "青禾小组如何完成土壤调查？")
    [decision] = accepted(decisions)
    assert decision["fields"]["method"]["text"] == "分区采样"
    assert decision["fields"]["subject"]["text"] == "青禾小组"
    assert decision["fields"]["effect"]["text"] == "完成土壤调查"
    assert decision["statement_kind"] == "fact"
    exact(index, decision)


def test_generic_operation_query_can_bind_explicit_source_goal():
    index, _, _, decisions = discourse("工作人员通过清理淤泥疏通河道。", "如何疏通河道？")
    [decision] = accepted(decisions)
    assert decision["fields"]["method"]["text"] == "清理淤泥"
    exact(index, decision)


@pytest.mark.parametrize("source,question,primary", [
    ("青禾小组通过模拟实验完成验证。", "青禾小组如何完成验证？", "模拟实验"),
    ("青禾小组通过阅读说明书完成安装。", "青禾小组如何完成安装？", "阅读说明书"),
    ("青禾小组通过假设检验完成分析。", "青禾小组如何完成分析？", "假设检验"),
    ("由于使用说明不清，所以设备损坏。", "设备为何损坏？", "使用说明不清"),
    ("因为准备不足，所以青禾项目延期启动。", "青禾项目为何延期启动？", "准备不足"),
])
def test_status_guard_does_not_match_inside_factual_technical_words(source, question, primary):
    index, result, _, decisions = discourse(source, question)
    [decision] = accepted(decisions)
    assert decision["fields"][result["question"]["type"]]["text"] == primary
    exact(index, decision)


@pytest.mark.parametrize("source", [
    "要完成土壤调查，可以通过它。",
    "土壤调查的方法是使用它。",
    "为了完成土壤调查，青禾小组通过它。",
])
def test_method_never_invents_ambiguous_referred_instrument(source):
    _, _, _, decisions = discourse(source, "如何完成土壤调查？")
    assert not accepted(decisions)


def test_method_definition_is_a_source_procedure_not_a_completed_fact():
    index, _, _, decisions = discourse("青禾小组完成土壤调查的方法是分区采样。", "青禾小组如何完成土壤调查？")
    [decision] = accepted(decisions)
    assert decision["fields"]["method"]["text"] == "分区采样"
    assert decision["fields"]["target"]["text"] == "青禾小组完成土壤调查"
    assert decision["statement_kind"] == "procedure"
    exact(index, decision)


def test_explicit_instruction_keeps_modality_and_original_prescription():
    source = "要完成土壤调查，可以通过分区采样。"
    index, _, _, decisions = discourse(source, "怎样才能完成土壤调查？")
    [decision] = accepted(decisions)
    assert decision["fields"]["method"]["text"] == "分区采样"
    assert decision["fields"]["effect"]["text"] == "完成土壤调查"
    assert decision["fields"]["modality"]["text"] == "可以"
    assert decision["fields"]["prescription"]["text"] == source
    assert decision["statement_kind"] == "procedure"
    exact(index, decision)


@pytest.mark.parametrize("source", [
    "青禾小组可以通过分区采样完成土壤调查。",
    "青禾小组通过分区采样可以完成土壤调查。",
])
def test_inline_source_modal_is_reported_as_procedure(source):
    index, _, _, decisions = discourse(source, "青禾小组怎样完成土壤调查？")
    [decision] = accepted(decisions)
    assert decision["fields"]["method"]["text"] == "分区采样"
    assert decision["statement_kind"] == "procedure"
    exact(index, decision)


@pytest.mark.parametrize("source,question", [
    ("松林小组通过分区采样完成土壤调查。", "青禾小组如何完成土壤调查？"),
    ("周宁通过鼓励帮助林青。", "林青如何帮助周宁？"),
    ("青禾小组通过分区采样完成水质调查。", "青禾小组如何完成土壤调查？"),
    ("青禾小组通过分区采样完成土壤调查。", "青禾小组如何独立完成土壤调查？"),
    ("青禾小组计划通过分区采样完成土壤调查。", "青禾小组如何完成土壤调查？"),
    ("它通过分区采样完成土壤调查。", "如何完成土壤调查？"),
    ("小红通过演示要求小明完成土壤调查。", "小明如何完成土壤调查？"),
    ("小明通过演示要求小红完成土壤调查。", "小明如何完成土壤调查？"),
])
def test_method_role_scope_modifiers_and_nonfactual_guards(source, question):
    _, _, _, decisions = discourse(source, question)
    assert not accepted(decisions)


@pytest.mark.parametrize("separator", ["", "\n", "\n\n", "\r\n"])
@pytest.mark.parametrize("reference", ["原因是", "这是因为"])
def test_adjacent_reason_has_exact_antecedent_and_real_binding(separator, reference):
    source = "\ufeff　青禾展馆暂停开放。　" + separator + reference + "入口处的水管破裂。"
    index, _, _, decisions = discourse(source, "青禾展馆为什么暂停开放？", max_chars=16)
    [decision] = accepted(decisions)
    assert decision["rule"] == "D104"
    assert decision["fields"]["reason"]["text"] == "入口处的水管破裂"
    assert decision["fields"]["reference"]["text"] == reference
    assert decision["fields"]["effect"]["sentence_id"] != decision["fields"]["reason"]["sentence_id"]
    assert decision["binding"]["antecedent"] == decision["fields"]["effect"]
    assert not decision["binding"]["checks"]["boundary_reasons"]
    exact(index, decision)


@pytest.mark.parametrize("source", [
    "青禾展馆暂停开放。天气晴朗。原因是入口受损。",
    "青禾展馆暂停开放。\n第二章 设备\n原因是入口受损。",
    "青禾展馆暂停开放。\n1. 原因是入口受损。",
    "青禾展馆暂停开放，松林项目开始。原因是入口受损。",
    "青禾展馆没有暂停开放。原因是入口受损。",
    "该展馆暂停开放。原因是入口受损。",
    "青禾展馆暂停开放。原因是它受损。",
])
def test_adjacent_reason_never_crosses_boundary_or_guesses_reference(source):
    _, _, _, decisions = discourse(source, "青禾展馆为什么暂停开放？")
    assert not accepted(decisions)


def test_adjacency_requires_both_eligible_and_obeys_frozen_boundary():
    index, result, evidence, decisions = discourse("青禾展馆暂停开放。原因是入口受损。", "青禾展馆为何暂停开放？")
    assert len(accepted(decisions)) == 1
    assert not accepted(extract_discourse(index, result, {"candidates": evidence["candidates"][:1]}))
    left, right = evidence["candidates"]
    index["context"]["boundaries"].append({"left_sentence_id": left["sentence_id"],
        "right_sentence_id": right["sentence_id"], "reasons": ["CW004"]})
    assert not accepted(extract_discourse(index, result, evidence))


def test_named_procedure_steps_remain_one_real_source_sentence():
    index, _, _, decisions = discourse("土壤调查的方法如下。首先划分区域，然后分区采样，最后记录数据。", "土壤调查如何独立进行？")
    # An extra requested qualifier must not silently disappear.  Exact target
    # queries below are accepted.
    assert not accepted(decisions)
    index, _, _, decisions = discourse("土壤调查的方法如下。首先划分区域，然后分区采样，最后记录数据。", "土壤调查用什么方法？")
    [decision] = accepted(decisions)
    assert decision["rule"] == "D204"
    assert decision["statement_kind"] == "procedure"
    assert decision["fields"]["method"]["text"] == "首先划分区域，然后分区采样，最后记录数据。"
    exact(index, decision)


def test_actual_token_boundaries_prohibit_substring_target_match():
    _, _, _, decisions = discourse("因为入口受损，所以青禾展馆暂停开放。", "青禾展馆为何暂停开放？",
                                   book_words=["青禾展馆暂停开放"])
    assert not accepted(decisions)


def test_completed_query_state_requires_explicit_completed_source_marker():
    _, _, _, decisions = discourse("因为道路积水，所以青禾项目启动。", "青禾项目为什么已经启动？")
    assert not accepted(decisions)
    index, _, _, decisions = discourse("因为道路积水，所以青禾项目已经启动。", "青禾项目为什么已经启动？")
    [decision] = accepted(decisions)
    assert decision["fields"]["effect"]["text"] == "青禾项目已经启动"
    exact(index, decision)


def test_rules_are_serializable_and_inputs_do_not_change():
    index, result, evidence, _ = discourse("因为道路积水，所以青禾项目延期启动。", "青禾项目为何延期启动？")
    before = deepcopy((index, result, evidence))
    extract_discourse(index, result, evidence)
    assert (index, result, evidence) == before
    assert DISCOURSE_POLICY and DISCOURSE_CONFIG["max_sentences"] == 2
    assert {"D101", "D104", "D201", "D203", "DG101", "DG102", "DG103"} <= DISCOURSE_RULES.keys()
    assert "cause_direction" in DISCOURSE_PATTERNS
