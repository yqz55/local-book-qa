"""End-to-end risk checks for source-bound, deterministic first-version answers."""

import copy
import math
from pathlib import Path

import pytest

from evidence import ENGINE_VERSION, EXPANSION_ENGINE_VERSION
from evidence.dag_tokenization import default_lexicon, lexicon_snapshot
from evidence.ingest import IngestConfig
from evidence.query_expansion import resource_snapshot
from evidence.retrieval import build_index, digest, search
from evidence.tracing import deserialize, make_trace, read_record, replay, serialize


EMPTY_WORDNET = resource_snapshot([], {"name": "controlled-answer-tests"})


def corpus(text, *, max_chars=400, engine=None, book_words=None):
    kwargs = {"expansion_resource": EMPTY_WORDNET, "lexicon": default_lexicon()}
    if engine:
        kwargs["_engine_version"] = engine
    return build_index(text.encode("utf-8"), "短文规则验收.txt", "utf-8", [], IngestConfig(max_chars), book_words=book_words, **kwargs)


def qa(text, question, **kwargs):
    index = corpus(text, **kwargs)
    return index, search(index, question)


def assert_exact_fields(index, answer):
    raw, clean = index["source"]["text"], index["cleaning"]["text"]
    sentences = {s["id"]: s for s in index["sentences"]["sentences"]}
    for field in answer["fields"].values():
        assert raw[field["source_start"]:field["source_end"]] == field["text"]
        # Clean/source spans may differ in whitespace, but both must stay inside
        # the very sentence cited for the extracted fact.
        sentence = sentences[field["sentence_id"]]
        assert sentence["source_start"] <= field["source_start"] < field["source_end"] <= sentence["source_end"]
        assert sentence["clean_start"] <= field["clean_start"] < field["clean_end"] <= sentence["clean_end"]
        assert clean[field["clean_start"]:field["clean_end"]]


def test_definition_has_exact_original_text_after_cleaning_and_block_split():
    text = "\ufeff　光合作用是绿色植物把光能转化为化学能的过程。　\r\n另一个段落讨论空气。"
    index, result = qa(text, "请问光合作用是什么？", max_chars=16)
    answer = result["answer"]
    assert answer["status"] == "answered"
    assert answer["fields"]["definition"]["text"] == "光合作用是绿色植物把光能转化为化学能的过程。"
    assert len(result["sentence_evidence"]["candidates"][0]["paragraph_ids"]) > 1
    assert_exact_fields(index, answer)


def test_definition_may_use_jiang_to_describe_a_process():
    source = "雨水收集是将降雨储存后用于灌溉的方法。"
    index, result = qa(source, "雨水收集是什么？", book_words=["雨水收集"])
    assert result["answer"]["status"] == "answered"
    assert result["answer"]["fields"]["definition"]["text"] == source
    assert_exact_fields(index, result["answer"])


def test_jiang_gift_grammar_is_not_mistaken_for_future_modality():
    index, result = qa("林青将一本书送给周宁。", "谁将一本书送给周宁？")
    assert result["answer"]["status"] == "answered"
    assert result["answer"]["fields"]["value"]["text"] == "林青"
    assert result["answer"]["fields"]["object"]["text"] == "一本书"
    assert result["answer"]["fields"]["recipient"]["text"] == "周宁"
    assert_exact_fields(index, result["answer"])


def test_question_words_never_score_or_expand_and_actual_content_still_matches():
    index, result = qa("火星探测器于2020年7月23日成功发射。\n为什么如何什么时候这些词是疑问表达。", "请问火星探测器是什么时候发射？")
    assert result["question"]["type"] == "time"
    assert result["answer"]["status"] == "answered"
    assert result["answer"]["fields"]["time"]["text"] == "2020年7月23日"
    filtered = {"请问", "是什么时候", "什么时候", "时候", "什么", "是"}
    assert not filtered.intersection(t["term"] for t in result["query_terms"])
    assert not filtered.intersection(e.get("source_term") for e in result["lexical_expansions"])
    assert result["top_ids"][0] == "P000001"
    assert_exact_fields(index, result["answer"])


def test_which_year_question_does_not_invent_an_explicit_year_constraint():
    index, result = qa("项目于2025年启动。", "项目哪一年启动？")
    assert not result["question"]["constraints"]
    assert result["answer"]["status"] == "answered"
    assert result["answer"]["fields"]["time"]["text"] == "2025年"
    assert_exact_fields(index, result["answer"])


@pytest.mark.parametrize("number", ["3.14克", "10至12克", "约3.14克", "不超过10克", "3.14%"])
def test_quantity_keeps_decimal_range_approximation_and_unit(number):
    index, result = qa(f"实验样品的重量为{number}。", "实验样品的重量是多少？")
    assert result["answer"]["status"] == "answered"
    assert result["answer"]["fields"]["value"]["text"] == number
    assert_exact_fields(index, result["answer"])


def test_gift_role_reversal_does_not_supply_the_wrong_subject():
    index, result = qa("王明把雨伞送给李华。\n李华把雨伞送给王明。", "谁把雨伞送给王明？")
    assert result["answer"]["status"] == "answered"
    assert result["answer"]["fields"]["value"]["text"] == "李华"
    assert result["answer"]["fields"]["recipient"]["text"] == "王明"
    assert any(not d["accepted"] and "recipient" in d["reason"] for d in result["answer"]["decisions"])
    assert_exact_fields(index, result["answer"])


@pytest.mark.parametrize("source", [
    "李华没有把雨伞送给王明。",
    "李华未把雨伞送给王明。",
    "李华不把雨伞送给王明。",
    "李华计划把雨伞送给王明。",
    "如果李华把雨伞送给王明，他会道谢。",
    "据说李华把雨伞送给王明。",
    "李华说：“我把雨伞送给王明。”",
])
def test_nonfactual_or_reported_gifts_do_not_become_facts(source):
    _, result = qa(source, "谁把雨伞送给王明？")
    assert result["sentence_evidence"]["candidates"]
    assert result["answer"]["status"] != "answered"
    assert not result["answer"]["fields"]


@pytest.mark.parametrize("question", ["谁没有把雨伞送给王明？", "谁计划把雨伞送给王明？", "它何时发射？"])
def test_question_fact_constraints_stop_unsupported_short_answers(question):
    _, result = qa("李华把雨伞送给王明。\n它于2020年7月23日发射。", question)
    assert result["question"]["constraints"]
    assert result["answer"]["status"] != "answered"


def test_known_event_must_bind_to_date_in_same_clause():
    _, result = qa("火星探测器成功发射，另一项目于2020年7月23日启动。", "火星探测器何时发射？")
    assert result["sentence_evidence"]["candidates"]
    assert result["answer"]["status"] != "answered"


@pytest.mark.parametrize("source, question", [
    ("乙于2024年击败甲。", "甲何时击败乙？"),
    ("甲帮助乙于2024年发射。", "甲何时发射？"),
])
def test_time_answer_requires_event_order_and_no_mediating_subject(source, question):
    _, result = qa(source, question)
    assert result["sentence_evidence"]["candidates"]
    assert result["answer"]["status"] != "answered"


def test_literal_anchor_cannot_match_inside_a_larger_protected_word():
    _, result = qa("甲馆于2024年发射。", "甲何时发射？", book_words=["甲馆"])
    assert result["sentence_evidence"]["candidates"]
    assert result["answer"]["status"] != "answered"


def test_quantity_unit_explicitly_requested_cannot_be_ignored():
    _, result = qa("实验样品重量为300升。", "实验样品重量是多少克？")
    assert result["sentence_evidence"]["candidates"]
    assert result["answer"]["status"] != "answered"


@pytest.mark.parametrize("quantity", ["不到10克", "不低于10克", "低于10克", "大于10克", "10克以上", "12至10克", "10克到12克"])
def test_quantity_never_drops_comparison_or_reorders_range(quantity):
    _, result = qa(f"实验样品重量为{quantity}。", "实验样品重量是多少？")
    answer = result["answer"]
    assert answer["status"] != "answered" or answer["fields"]["value"]["text"] == quantity


@pytest.mark.parametrize("source,question", [
    ("火星探测器于2020年7月23日和2021年2月10日发射。", "火星探测器何时发射？"),
    ("实验样品重量为3克和5克。", "实验样品重量是多少？"),
])
def test_multiple_unbound_values_do_not_pick_first(source, question):
    _, result = qa(source, question)
    assert result["answer"]["status"] == "ambiguous"
    assert not result["answer"]["fields"]


def test_target_conflicting_dates_are_exposed_without_selecting_first():
    _, result = qa("火星探测器于2020年7月23日发射。\n火星探测器于2021年2月10日发射。", "火星探测器何时发射？")
    assert result["answer"]["status"] == "ambiguous"
    assert len(result["answer"]["evidence_ids"]) == 2
    assert not result["answer"]["fields"]


@pytest.mark.parametrize("question", ["为什么？", "为什么会这样？", "请问什么时候？"])
def test_missing_content_is_not_retrieved_using_question_words(question):
    _, result = qa("为什么如何什么时候都是疑问表达。", question)
    assert result["question"]["status"] == "no_content"
    assert result["query_terms"] == []
    assert result["candidates"] == []
    assert result["answer"]["status"] == "no_content"


def test_reason_method_and_mixed_slots_keep_evidence_without_claiming_answer():
    index = corpus("火星探测器于2020年7月23日成功发射。")
    for question in ("火星探测器为什么发射？", "火星探测器怎样发射？", "火星探测器什么时候发射，重量是多少？"):
        result = search(index, question)
        assert result["sentence_evidence"]["candidates"]
        assert result["answer"]["status"] == "evidence_only"
        assert result["answer"]["mode"] == "reference"
        assert not result["answer"]["fields"]


@pytest.fixture(scope="module")
def answer_trace():
    index = build_index("光合作用是绿色植物把光能转化为化学能的过程。".encode(), "复算短文.txt", "utf-8", ["光合作用"],
                        lexicon=lexicon_snapshot({}), expansion_resource=EMPTY_WORDNET)
    result = search(index, "光合作用是什么？")
    assert result["answer"]["status"] == "answered"
    return make_trace(index, result)


@pytest.mark.parametrize("target", ["question_decision", "sentence_boundary", "sentence_score", "answer_span", "template"])
def test_rehashed_trace_tamper_is_still_detected(answer_trace, target):
    # Real saved records cannot preserve aliases between an answer's template
    # view and the template resource; round-trip before modifying either copy.
    trace = deserialize(serialize(answer_trace))
    result = trace["retrieval"]
    if target == "question_decision":
        result["question"]["decisions"][0]["reason"] = "伪造规则解释"
    elif target == "sentence_boundary":
        trace["index"]["sentences"]["sentences"][0]["source_end"] -= 1
    elif target == "sentence_score":
        result["sentence_evidence"]["candidates"][0]["score"] += 0.01
    elif target == "answer_span":
        result["answer"]["fields"]["definition"]["source_end"] -= 1
    elif target == "template":
        result["answer"]["template"]["text"] = "伪造结论：{definition}"
    trace["integrity_sha256"] = digest({k: v for k, v in trace.items() if k != "integrity_sha256"})
    with pytest.raises(ValueError, match="复算"):
        replay(trace)


def test_full_question_sentence_answer_replay_succeeds(answer_trace):
    report = replay(answer_trace)
    assert report["verified"] is True
    assert any("问题提炼" in check for check in report["checks"])
    assert any("字段来源" in check for check in report["checks"])


def test_historical_15_trace_replays_with_original_question_scoring():
    # A deliberately tiny frozen historical snapshot demonstrates the old
    # behavior independently of new default question/sentence resources.
    index = build_index("为什么\n甲".encode(), "旧版.txt", "utf-8", [],
                        lexicon=lexicon_snapshot({}), expansion_resource=EMPTY_WORDNET,
                        _engine_version=EXPANSION_ENGINE_VERSION)
    result = search(index, "为什么", _engine_version=EXPANSION_ENGINE_VERSION)
    assert "question" not in result and "answer" not in result
    assert [term["term"] for term in result["query_terms"]] == ["为", "什", "么"]
    assert result["candidates"][0]["score"] == pytest.approx(3 * math.log(2) * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 3 / 2)))
    trace = make_trace(index, result)
    assert trace["schema"] == "evidence-trace/6"
    assert trace["engine_version"] == EXPANSION_ENGINE_VERSION
    assert replay(trace)["verified"] is True
    current = search(index, "为什么")
    assert current["question"]["status"] == "no_content"
    assert current["answer"]["status"] == "no_content"


def test_frozen_15_file_preserves_old_scoring_and_upgrades_without_changing_statistics():
    old = read_record(Path(__file__).parent / "fixtures" / "query-expansion-v15-trace.json")
    before = copy.deepcopy(old)
    assert old["engine_version"] == EXPANSION_ENGINE_VERSION
    assert old["schema"] == "evidence-trace/6"
    assert replay(old)["verified"] is True
    assert "question" not in old["retrieval"]
    assert "answering" not in old["index"] and "sentences" not in old["index"]
    assert [t["term"] for t in old["retrieval"]["query_tokens"]] == ["计算机", "是", "什么"]
    old_question_match = next(c for c in old["retrieval"]["candidates"] if c["paragraph_id"] == "P000002")
    assert any(t["term"] == "什么" and t["contribution"] > 0 for t in old_question_match["term_details"])

    exported = make_trace(old["index"], old["retrieval"])
    assert exported["engine_version"] == EXPANSION_ENGINE_VERSION
    assert exported["schema"] == "evidence-trace/6"
    assert exported["retrieval"] == old["retrieval"]
    assert replay(exported)["verified"] is True

    current = search(old["index"], old["retrieval"]["query"])
    assert current["question"]["type"] == "definition"
    assert current["question"]["target"] == "计算机"
    assert current["question"]["keywords"] == ["计算机"]
    assert {t["term"] for t in current["query_terms"]} == {"计算机", "电脑"}
    assert current["top_ids"] == ["P000001"]
    upgraded = make_trace(old["index"], current)
    assert upgraded["engine_version"] == ENGINE_VERSION
    assert upgraded["schema"] == "evidence-trace/10"
    assert upgraded["index"]["statistics"] == old["index"]["statistics"]
    assert upgraded["index"]["postings"] == old["index"]["postings"]
    assert upgraded["index"]["source"] == old["index"]["source"]
    assert replay(upgraded)["verified"] is True
    assert old == before
