"""General vocabulary, real DAG decisions, same analyzer, frozen replay."""

import copy
from pathlib import Path

import pytest

from evidence import ENGINE_VERSION, PERSON_ENGINE_VERSION
from evidence.dag_tokenization import DAG_POLICY, DAGTokenizer, default_lexicon, lexicon_snapshot
from evidence.retrieval import SEGMENTATION_CONFIG, build_index, digest, rebuild_with_book_words, search
from evidence.tracing import make_trace, read_record, replay


def corpus(text, **kwargs):
    return build_index(text.encode(), "整句分词.txt", "utf-8", [], **kwargs)


def terms(result):
    return [t["term"] for t in result["query_tokens"]]


def test_general_dictionary_and_everyday_words_work_without_custom_words():
    index = corpus("我们一起结婚吧。\n我们知道夫妻不能一起去。")
    assert len(index["segmentation"]["lexicon"]["frequencies"]) > 300000
    assert index["dictionary"]["base_words"] == []
    assert index["dictionary"]["book_words"] == []
    assert terms(search(index, "我们一起结婚吧")) == ["我们", "一起", "结婚", "吧"]
    assert [t["term"] for t in index["paragraphs"][0]["tokens"]] == ["我们", "一起", "结婚", "吧"]
    assert {"我们", "知道", "夫妻", "不能", "一起"}.issubset(index["statistics"]["collection_tf"])


@pytest.mark.parametrize("sentence,expected", [
    ("今天下午我们去图书馆学习。", {"今天下午", "我们", "图书馆", "学习"}),
    ("程序员正在设计数据库。", {"程序员", "正在", "设计", "数据库"}),
    ("科学家研究生命的起源。", {"科学家", "研究", "生命", "起源"}),
    ("天气预报说明明天可能下雨。", {"天气预报", "说明", "明天", "可能", "下雨"}),
    ("计算机需要处理大量数据。", {"计算机", "需要", "处理", "大量", "数据"}),
])
def test_vocabulary_is_general_not_a_sentence_allowlist(sentence, expected):
    index = corpus(sentence)
    assert expected.issubset(terms(search(index, sentence)))
    assert all(t["rule"] != "D002" for t in search(index, sentence)["query_tokens"])


def test_plain_chinese_baseline_matches_jieba_exact_mode_without_hmm():
    import jieba
    resource = default_lexicon()
    analyzer = DAGTokenizer(resource, [], book_bonus_cap=0)
    reference = jieba.Tokenizer()
    reference.initialize()
    for text in ("我们一起结婚吧", "研究生命起源", "南京市长江大桥", "他来到了北京清华大学", "中华人民共和国成立了"):
        actual = [t["term"] for t in analyzer.tokenize(text)[0]]
        assert actual == reference.lcut(text, HMM=False)


def test_whole_path_resolves_an_ambiguity_that_longest_matching_gets_wrong():
    resource = lexicon_snapshot({"研究": 100, "研究生": 10, "生命": 100, "命": 1, "起源": 100})
    analyzer = DAGTokenizer(resource, [], book_bonus_cap=0)
    tokens, _ = analyzer.tokenize("研究生命起源", capture=True)
    assert [t["term"] for t in tokens] == ["研究", "生命", "起源"]
    node = analyzer.last_trace[0]["nodes"][0]
    assert {c["term"] for c in node["candidates"]} == {"研究", "研究生"}
    winner = node["candidates"][0]
    assert winner["term"] == "研究"
    assert winner["path_score"] == winner["base_score"] + winner["book_bonus"] + winner["suffix_score"]


def test_high_book_frequency_has_a_bounded_bonus_instead_of_forcing_a_short_prefix():
    resource = lexicon_snapshot({"自然": 1000, "自然语言处理": 1000, "语言": 1, "处理": 1})
    analyzer = DAGTokenizer(resource, [], seed_frequencies={"自然": 10**9}, book_bonus_cap=0.25)
    tokens, _ = analyzer.tokenize("自然语言处理", capture=True)
    assert [t["term"] for t in tokens] == ["自然语言处理"]
    assert all(0 <= c["book_bonus"] <= 0.25 for n in analyzer.last_trace[0]["nodes"] for c in n["candidates"])


def test_names_and_custom_terms_are_whole_and_document_tf_is_not_duplicated():
    index = corpus("赵清岚说道：好。\n清岚点头。\n量子相位校验模块。", book_words=["量子相位校验模块"])
    assert terms(search(index, "赵清岚与量子相位校验模块")) == ["赵清岚", "与", "量子相位校验模块"]
    assert {t["term"] for t in search(index, "赵清岚")["query_terms"]} == {"赵清岚", "清岚"}
    assert index["paragraphs"][0]["term_frequencies"].get("清岚", 0) == 0
    assert index["statistics"]["collection_tf"]["清岚"] == 1
    assert index["paragraphs"][2]["tokens"][0]["rule"] == "D005"


def test_normal_single_character_words_are_distinct_from_unknown_character_fallback():
    index = corpus("我𰻞。")
    tokens = search(index, "我𰻞")["query_tokens"]
    assert [(t["term"], t["rule"]) for t in tokens] == [("我", "D001"), ("𰻞", "D002")]


def test_source_coordinates_and_actual_document_query_tokenization_agree():
    text = "\ufeff　我们一起结婚吧\tABC-12 😀 α +！\r\n赵清岚说道：好。\r\n清岚点头。"
    index = corpus(text)
    result = search(index, text)
    assert [t["term"] for p in index["paragraphs"] for t in p["tokens"]] == terms(result)
    for token in result["query_tokens"]:
        assert text[token["source_start"]:token["source_end"]] == token["surface"]
    for p in index["paragraphs"]:
        for token in p["tokens"]:
            assert text[token["source_start"]:token["source_end"]] == token["surface"]
    assert replay(make_trace(index, result))["verified"]


def test_book_adaptation_is_frozen_and_queries_do_not_change_the_index():
    resource = lexicon_snapshot({"我们": 200, "一起": 100, "结婚": 50, "吧": 10})
    index = corpus("我们一起结婚吧。\n我们一起。", lexicon=resource)
    before = copy.deepcopy(index)
    for query in ("结婚", "一起一起", "我们一起结婚吧"):
        assert search(index, query)["query_tokenization"]["hmm"] is False
    assert index == before
    updated = rebuild_with_book_words(index, ["我们一起"])
    assert updated["segmentation"]["lexicon"] == resource
    assert updated["segmentation"]["config"] == index["segmentation"]["config"]
    assert updated["index_id"] != index["index_id"]
    assert replay(make_trace(updated, search(updated, "我们一起")))["verified"]


@pytest.mark.parametrize("tamper", ["path", "seed", "lexicon"])
def test_replay_recomputes_decisions_and_rejects_tampering_even_after_resigning(tamper):
    resource = lexicon_snapshot({"我们": 20, "一起": 10})
    index = corpus("我们一起。", lexicon=resource)
    trace = make_trace(index, search(index, "我们一起"))
    if tamper == "path":
        trace["retrieval"]["query_segmentation_trace"][0]["nodes"][0]["candidates"][0]["path_score"] += 1
    elif tamper == "seed":
        trace["index"]["segmentation"]["seed_collection_tf"]["我们"] += 1
    else:
        trace["index"]["segmentation"]["lexicon"]["frequencies"]["我们"] += 1
    trace["integrity_sha256"] = digest({k: v for k, v in trace.items() if k != "integrity_sha256"})
    with pytest.raises(ValueError):
        replay(trace)


def test_current_snapshot_replays_without_loading_the_installed_dictionary(monkeypatch):
    import evidence.retrieval as retrieval
    index = corpus("我们一起结婚吧。")
    trace = make_trace(index, search(index, "我们一起结婚吧"))
    monkeypatch.setattr(retrieval, "default_lexicon", lambda: pytest.fail("Snapshot must supply its own dictionary"))
    assert replay(trace)["verified"]


def test_frozen_person_version_still_replays_and_can_be_upgraded():
    old = read_record(Path(__file__).parent / "fixtures" / "person-v13-trace.json")
    assert old["engine_version"] == PERSON_ENGINE_VERSION
    assert replay(old)["verified"]
    result = search(old["index"], "我们一起结婚吧")
    assert terms(result) == ["我们", "一起", "结婚", "吧"]
    trace = make_trace(old["index"], result)
    assert trace["engine_version"] == ENGINE_VERSION
    assert trace["retrieval"]["query_tokenization"]["policy"] == DAG_POLICY
    assert replay(trace)["verified"]


@pytest.mark.parametrize("cap", [-1, 2, float("nan"), True])
def test_invalid_bonus_configuration_rejected(cap):
    cfg = {**SEGMENTATION_CONFIG, "book_bonus_cap": cap}
    with pytest.raises(ValueError):
        corpus("我们一起。", lexicon=lexicon_snapshot({}), segmentation_config=cfg)
