"""Question understanding must improve retrieval without changing source spans."""

import copy

import pytest

from evidence.dag_tokenization import DAGTokenizer, default_lexicon, lexicon_snapshot
from evidence.ingest import SourceMap, clean_text
from evidence.questions import analyze_question


def plan(raw, *, words=(), names=(), book_words=(), empty_lexicon=False):
    cleaned = clean_text(raw)
    tokenizer = DAGTokenizer(lexicon_snapshot({}) if empty_lexicon else default_lexicon(),
                             words, names=names, book_words=book_words)
    tokens, _ = tokenizer.tokenize(cleaned["text"], mapping=SourceMap(cleaned["source_runs"], raw))
    before = copy.deepcopy(tokens)
    result = analyze_question(cleaned, raw, tokens)
    assert tokens == before  # Parsing must not mutate the segmentation trace.
    for token in result["retrieval_tokens"]:
        assert any(token is original for original in tokens)
    for item in result["decisions"] + result["constraints"] + result["type_matches"]:
        assert cleaned["text"][item["clean_start"]:item["clean_end"]] == item["text"]
        assert raw[item["source_start"]:item["source_end"]] == item["text"]
    return result, tokens


@pytest.mark.parametrize("query, kind, target", [
    ("光合作用是什么？", "definition", "光合作用"),
    ("请问什么是光合作用？", "definition", "光合作用"),
    ("光合作用的定义是什么？", "definition", "光合作用"),
    ("天问一号什么时候发射？", "time", "天问一号发射"),
    ("天问一号是什么时候发射？", "time", "天问一号发射"),
    ("研究站有多少成员？", "quantity", "研究站成员"),
    ("水为什么会蒸发？", "reason", "水蒸发"),
    ("植物如何吸收水分？", "method", "植物吸收水分"),
    ("文章如何描述光合作用？", "description", "光合作用"),
])
def test_question_type_and_literal_target(query, kind, target):
    result, _ = plan(query)
    assert result["type"] == kind
    assert result["target"] == target
    assert result["status"] == "ok"
    assert not set(result["keywords"]) & {"什么", "为什么", "如何", "什么时候", "多少", "请问", "是", "的"}


def test_long_question_expression_is_removed_even_with_character_fallback():
    result, tokens = plan("请告诉我天问一号在什么时候发射？", empty_lexicon=True)
    assert result["type"] == "time"
    assert result["target"] == "天问一号发射"
    assert "时" not in result["keywords"]
    assert "候" not in result["keywords"]
    assert len(result["retrieval_tokens"]) < len(tokens)


def test_negation_and_plan_are_constraints_not_query_score_terms():
    result, _ = plan("宝玉为什么原想把玉送给黛玉，后来没有送出？", names=("宝玉", "黛玉"))
    assert result["type"] == "reason"
    assert {(x["kind"], x["value"]) for x in result["constraints"]} >= {("negation", "negative"), ("status", "planned")}
    assert not set(result["keywords"]) & {"没有", "原想", "把", "为什么"}
    assert any(d["rule"] == "QP004" and d["text"] == "没有" for d in result["decisions"])


def test_explicit_time_and_location_are_retained_with_coordinates():
    result, _ = plan("2020年研究所在北京成立了多少个实验室？")
    assert any(c["kind"] == "time" and c["text"] == "2020年" for c in result["constraints"])
    assert any(c["kind"] == "location" and c["text"] == "北京" for c in result["constraints"])
    assert any("北京" in term for term in result["keywords"])
    assert any("2020" in term for term in result["keywords"])


def test_interrogative_year_and_location_are_not_explicit_constraints():
    result, _ = plan("项目哪一年启动？")
    assert result["type"] == "time"
    assert not result["constraints"]
    result, _ = plan("会议在什么时候举办？")
    assert not any(c["kind"] == "location" for c in result["constraints"])


@pytest.mark.parametrize("query, expected", [
    ("谁把玉送给黛玉？", {"subject": None, "object": "玉", "recipient": "黛玉", "requested_role": "subject"}),
    ("请问书中谁把玉送给黛玉？", {"subject": None, "object": "玉", "recipient": "黛玉", "requested_role": "subject"}),
    ("宝玉把什么送给黛玉？", {"subject": "宝玉", "object": None, "recipient": "黛玉", "requested_role": "object"}),
    ("宝玉将玉赠予谁？", {"subject": "宝玉", "object": "玉", "recipient": None, "requested_role": "recipient"}),
    ("谁送给李华一把雨伞？", {"subject": None, "object": "一把雨伞", "recipient": "李华", "requested_role": "subject"}),
    ("王明给李华送了什么？", {"subject": "王明", "object": None, "recipient": "李华", "requested_role": "object"}),
])
def test_gift_question_role_binding(query, expected):
    result, _ = plan(query, names=("宝玉", "黛玉", "王明", "李华"))
    assert result["type"] == "relation"
    assert result["relation"] is not None
    for field, value in expected.items():
        assert result["relation"][field] == value
    for role in result["relation"]["roles"].values():
        assert query[role["source_start"]:role["source_end"]] == role["text"]


def test_multiple_missing_gift_roles_are_not_treated_as_single_answer():
    result, _ = plan("谁把什么送给李华？", names=("李华",))
    assert result["type"] == "unsupported"
    assert result["relation"]["requested_role"] is None
    assert result["relation"]["question_role_count"] == 2


def test_question_with_multiple_answer_fields_is_not_silently_reduced():
    result, _ = plan("谁在什么时候把雨伞送给李华？", names=("李华",))
    assert result["type"] == "unsupported"
    assert result["question_types"] == ["relation", "time"]


def test_unresolved_pronouns_are_explicit_question_constraints():
    result, _ = plan("它何时发射？")
    assert any(c["kind"] == "reference" and c["text"] == "它" for c in result["constraints"])
    assert "它" in result["keywords"]
    result, _ = plan("“它”是什么意思？")
    assert not any(c["kind"] == "reference" for c in result["constraints"])


def test_quoted_interrogatives_are_literal_content():
    result, _ = plan("书中“为什么”一词出现在哪里？")
    assert "为什么" in "".join(result["keywords"])
    literal = [d for d in result["decisions"] if d["action"] == "retained" and d["rule"] == "QP002"]
    assert "为什么" in "".join(x["text"] for x in literal)
    assert not any(m["type"] == "reason" for m in result["type_matches"])


def test_names_and_book_terms_cannot_be_filtered_as_question_substrings():
    result, tokens = plan("何时清 什么山 十万个为什么", names=("何时清",), book_words=("什么山", "十万个为什么"))
    assert result["retrieval_tokens"] == tokens
    assert result["type"] == "keywords"
    assert result["keywords"] == ["何时清", "什么山", "十万个为什么"]
    result, _ = plan("《十万个为什么》是什么？")
    assert "为什么" in "".join(result["keywords"])
    assert result["type"] == "definition"


def test_no_substring_deletion_inside_an_unprotected_dictionary_term():
    result, tokens = plan("为什么学", words=("为什么学",), empty_lexicon=True)
    assert result["retrieval_tokens"] == tokens
    assert result["type"] == "keywords"


@pytest.mark.parametrize("query", ["甲甲乙", "甲的乙", "甲 是 乙", "ISBN ABC-12 3.14", "信息检索", "送给？", "林姑娘 没有 赠与"])
def test_plain_keywords_keep_existing_segmentation(query):
    result, tokens = plan(query)
    assert result["retrieval_tokens"] == tokens
    assert result["type"] == "keywords"


@pytest.mark.parametrize("query", ["为什么？", "什么？", "请问什么时候？", "怎样？", "谁？", "为什么会这样？", "这是什么？", "？？\n"])
def test_question_only_input_has_no_content(query):
    result, _ = plan(query)
    assert result["status"] == "no_content"
    assert result["retrieval_tokens"] == []
    assert result["target"] == ""


def test_quoted_deictics_remain_real_retrieval_objects():
    result, _ = plan("“这样”是什么意思？")
    assert result["status"] == "ok"
    assert "这样" in result["keywords"]


@pytest.mark.parametrize("auxiliary", ["要", "会", "能", "可以", "能够", "应该", "应当"])
def test_standalone_question_auxiliaries_are_status_constraints_not_noise(auxiliary):
    result, _ = plan(f"为什么{auxiliary}雨水收集？")
    assert auxiliary not in result["keywords"]
    assert any(c["kind"] == "status" and c["value"] == "modal" and c["text"] == auxiliary for c in result["constraints"])
    assert result["status"] == "ok"
    literal, _ = plan(f"“{auxiliary}”是什么意思？")
    assert auxiliary in "".join(literal["keywords"])
    plain, tokens = plan(f"雨水 {auxiliary} 收集")
    assert plain["retrieval_tokens"] == tokens


@pytest.mark.parametrize("context", ["书中", "文中", "文章中", "本文", "文献中"])
def test_document_context_prefix_does_not_constrain_sentence_content(context):
    result, _ = plan(f"请问{context}如何描述光合作用？")
    assert result["target"] == "光合作用"
    assert not any(context in term for term in result["keywords"])
    assert any(d["category"] == "document_context" and d["text"] == context and d["action"] == "filtered" for d in result["decisions"])
    literal, _ = plan(f"“{context}”是什么意思？")
    assert context in "".join(literal["keywords"])


def test_query_cleaning_offsets_and_parser_determinism():
    raw = "\ufeff　请问\t光合作用是什么？　\r\n"
    result, tokens = plan(raw)
    assert result["target"] == "光合作用"
    assert result == analyze_question(clean_text(raw), raw, tokens)
    assert next(d for d in result["decisions"] if d["text"] == "请问")["source_start"] == 2
