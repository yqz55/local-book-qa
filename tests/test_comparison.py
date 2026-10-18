"""A comparison must run two real pipelines over identical inputs."""

import copy

import pytest

from evidence import ANSWER_ENGINE_VERSION, ENGINE_VERSION
from evidence.comparison import build_previous_index, comparison_manifest
from evidence.dag_tokenization import default_lexicon
from evidence.query_expansion import resource_snapshot
from evidence.retrieval import BM25Config, build_index, configure_query_expansion, search
from evidence.tracing import make_trace, replay


def corpus(text):
    return build_index(text.encode(), "对比短文.txt", "utf-8", [], lexicon=default_lexicon(),
                       expansion_resource=resource_snapshot([], {"name": "comparison-tests"}))


def test_previous_pipeline_preserves_same_inputs_but_cannot_use_joint_rules():
    current = corpus("青禾项目进入实施阶段。\n该项目于2025年3月12日启动。")
    before = copy.deepcopy(current)
    previous = build_previous_index(current)
    query, config = "青禾项目什么时候启动？", BM25Config(k1=1.2, b=0.5)
    new = search(current, query, config)
    old = search(previous, query, config, _engine_version=ANSWER_ENGINE_VERSION)
    manifest = comparison_manifest(current, new, previous, old)
    assert all(manifest["checks"].values())
    assert old["answer"]["status"] == "evidence_only"
    assert new["answer"]["status"] == "answered"
    assert "window_retrieval" not in old and "context" not in previous
    assert previous["identity"]["engine_version"] == ANSWER_ENGINE_VERSION
    assert current["identity"]["engine_version"] == ENGINE_VERSION
    assert current == before
    assert replay(make_trace(previous, old))["verified"]
    assert replay(make_trace(current, new))["verified"]


def test_expansion_corrections_are_shared_and_mixed_queries_are_rejected():
    current = configure_query_expansion(corpus("林青把画册送给周宁。"), config={
        "enabled": False, "weight": 0.5, "max_alternatives": 2, "max_sources": 10}, blocked=["赠送"])
    previous = build_previous_index(current)
    old = search(previous, "画册", _engine_version=ANSWER_ENGINE_VERSION)
    new = search(current, "画册")
    assert comparison_manifest(current, new, previous, old)["checks"]["query_expansion"]
    assert previous["query_expansion"] == current["query_expansion"]
    with pytest.raises(ValueError, match="question"):
        comparison_manifest(current, new, previous, {**old, "query": "不同问题"})


def test_same_single_sentence_fact_remains_answered_in_both_versions():
    current = corpus("青禾项目于2025年启动。")
    previous = build_previous_index(current)
    new = search(current, "青禾项目何时启动？")
    old = search(previous, "青禾项目何时启动？", _engine_version=ANSWER_ENGINE_VERSION)
    assert old["answer"]["status"] == new["answer"]["status"] == "answered"
    assert old["answer"]["fields"]["time"]["text"] == new["answer"]["fields"]["time"]["text"] == "2025年"
