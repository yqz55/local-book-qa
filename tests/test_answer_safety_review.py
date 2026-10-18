"""Concrete independent-review regressions for unsupported factual answers.

The first release may refuse these sentences.  It must not silently remove a
negation, uncertainty, numeric sign, bound, era, or event/role distinction.
"""

from copy import deepcopy

import pytest

from evidence.dag_tokenization import lexicon_snapshot
from evidence.extraction import _digest, answering_snapshot
from evidence.retrieval import build_index, search


BASE_WORDS = [
    "项目", "启动", "成立", "验收", "人数", "预算", "箱子", "容量", "温度", "海拔", "变化率",
    "林清", "王明", "陈雨", "画册", "钢笔", "送给", "新闻", "日期", "档案", "报道",
]


def ask(text, query):
    index = build_index(text.encode("utf-8"), "独立审查短文.txt", "utf-8", BASE_WORDS,
                        lexicon=lexicon_snapshot({}))
    return search(index, query)


def assert_refused(result, question_type):
    assert result["question"]["type"] == question_type
    assert result["sentence_evidence"]["candidates"]
    assert result["answer"]["status"] != "answered", result["answer"]
    assert not result["answer"]["fields"]


def assert_preserved_or_refused(result, question_type, field_name, complete_expression):
    assert result["question"]["type"] == question_type
    assert result["sentence_evidence"]["candidates"]
    answer = result["answer"]
    if answer["status"] == "answered":
        assert answer["fields"][field_name]["text"] == complete_expression, answer
    else:
        assert not answer["fields"]


@pytest.mark.parametrize("text", [
    "项目于2025年没启动。",
    "项目于2025年未启动。",
    "项目于2025年不启动。",
    "项目于2025年取消启动。",
    "项目于2025年停止启动。",
    "项目或许于2025年启动。",
    "项目预计于2025年启动。",
    "项目拟在2025年启动。",
    "'项目于2025年启动。'",
])
def test_nonfactual_and_quoted_dates_do_not_become_completed_events(text):
    assert_refused(ask(text, "项目什么时候启动？"), "time")


@pytest.mark.parametrize("text,query", [
    ("项目启动后于2025年验收。", "项目什么时候启动？"),
    ("项目成立十年后于2025年启动。", "项目什么时候成立？"),
    ("项目启动的新闻在2025年发布。", "项目什么时候启动？"),
])
def test_date_of_another_event_cannot_answer_requested_event(text, query):
    assert_refused(ask(text, query), "time")


@pytest.mark.parametrize("text,expression", [
    ("项目于2025年前启动。", "2025年前"),
    ("项目于2025年之前启动。", "2025年之前"),
    ("项目于公元前2025年启动。", "公元前2025年"),
    ("项目在2020年代启动。", "2020年代"),
    ("项目于二〇二五年3月12日启动。", "二〇二五年3月12日"),
])
def test_temporal_bounds_era_decade_and_year_are_kept_or_refused(text, expression):
    assert_preserved_or_refused(ask(text, "项目什么时候启动？"), "time", "time", expression)


@pytest.mark.parametrize("text,query,expression", [
    ("人数约为20人。", "人数多少人？", "约为20人"),
    ("箱子容量至少为300升。", "箱子容量多少升？", "至少为300升"),
    ("温度为-5℃。", "温度多少？", "-5℃"),
    ("温度为零下5℃。", "温度多少？", "零下5℃"),
    ("海拔为负20米。", "海拔多少米？", "负20米"),
    ("变化率为-3%。", "变化率多少？", "-3%"),
])
def test_numeric_bounds_and_sign_are_kept_or_refused(text, query, expression):
    assert_preserved_or_refused(ask(text, query), "quantity", "value", expression)


@pytest.mark.parametrize("text,query", [
    ("林清昨天把画册送给王明。", "谁把画册送给王明？"),
    ("林清把画册送给王明后离开了。", "林清把画册送给谁？"),
    ("林清想把画册送给王明。", "谁把画册送给王明？"),
])
def test_gift_slot_does_not_absorb_adverb_predicate_or_intention(text, query):
    assert_refused(ask(text, query), "relation")


@pytest.mark.parametrize("negative", ["项目于2025年未启动。", "项目于2025年没启动。"])
def test_affirmative_and_negated_same_event_do_not_silently_choose_affirmative(negative):
    result = ask("项目于2025年启动。\n" + negative, "项目什么时候启动？")
    assert_refused(result, "time")


@pytest.mark.parametrize("text", ["{time:{missing}}", "{time:1000000}", "{time!r}"])
def test_template_rejects_nested_formats_alignment_and_conversions(text):
    snapshot = deepcopy(answering_snapshot())
    snapshot["templates"]["time"]["text"] = text
    snapshot["sha256"] = _digest({k: v for k, v in snapshot.items() if k != "sha256"})
    with pytest.raises(ValueError):
        answering_snapshot(snapshot)
