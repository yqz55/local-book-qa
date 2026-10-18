"""New discourse answers, frozen pipelines, and independent provenance replay."""

from pathlib import Path

import pytest

from evidence import ENGINE_VERSION, WINDOW_ENGINE_VERSION
from evidence.comparison import build_previous_index, comparison_manifest
from evidence.dag_tokenization import default_lexicon
from evidence.query_expansion import resource_snapshot
from evidence.retrieval import build_index, digest, search
from evidence.tracing import deserialize, make_trace, read_record, replay, serialize

RESOURCE = resource_snapshot([], {"name": "discourse-integration"})


def corpus(text):
    return build_index(text.encode(), "原因方法集成短文.txt", "utf-8", [],
                       lexicon=default_lexicon(), expansion_resource=RESOURCE)


@pytest.mark.parametrize("question,kind,target", [
    ("青禾项目延期启动的原因是什么？", "reason", "青禾项目延期启动"),
    ("青禾项目为什么会延期启动？", "reason", "青禾项目延期启动"),
    ("青禾小组完成土壤调查的方法是什么？", "method", "青禾小组完成土壤调查"),
    ("青禾小组怎样才能完成土壤调查？", "method", "青禾小组完成土壤调查"),
])
def test_natural_cause_and_method_questions_have_no_interrogative_content(question, kind, target):
    result = search(corpus("因为道路积水，所以青禾项目延期启动。\n青禾小组通过分区采样完成土壤调查。"), question)
    assert result["question"]["type"] == kind
    assert result["question"]["target"] == target
    assert not {"原因", "什么", "如何", "怎样", "才能", "会"}.intersection(result["question"]["keywords"])


@pytest.fixture(scope="module")
def trace():
    index = corpus("因为施工道路积水，所以青禾项目延期启动。")
    result = search(index, "青禾项目为什么会延期启动？")
    assert result["answer"]["status"] == "answered"
    assert result["answer"]["fields"]["reason"]["text"] == "施工道路积水"
    return make_trace(index, result)


@pytest.mark.parametrize("target", ["reason_span", "causal_direction", "statement_kind", "template", "policy"])
def test_rehashed_discourse_tampering_is_detected(trace, target):
    altered = deserialize(serialize(trace))
    answer = altered["retrieval"]["answer"]
    if target == "reason_span":
        answer["fields"]["reason"]["source_end"] -= 1
    elif target == "causal_direction":
        answer["fields"]["reason"]["text"] = "青禾项目延期启动"
    elif target == "statement_kind":
        answer["statement_kind"] = "procedure"
    elif target == "template":
        answer["text"] = "推测得出的新原因"
    else:
        altered["retrieval"]["discourse_policy"]["discourse_policy"] = "伪造关系"
    altered["integrity_sha256"] = digest({key: value for key, value in altered.items() if key != "integrity_sha256"})
    with pytest.raises(ValueError, match="复算"):
        replay(altered)


def test_new_cause_trace_and_both_frontend_versions_replay(trace):
    assert replay(trace)["verified"]
    assert any("因果" in check for check in replay(trace)["checks"])
    index = trace["index"]
    previous = build_previous_index(index)
    old_result = search(previous, trace["retrieval"]["query"], _engine_version=previous["identity"]["engine_version"])
    assert old_result["answer"]["status"] == "unsupported"
    assert all(comparison_manifest(index, trace["retrieval"], previous, old_result)["checks"].values())
    assert replay(make_trace(previous, old_result))["verified"]


def test_frozen_17_record_stays_unsupported_and_upgrades_to_source_backed_reason():
    old = read_record(Path(__file__).parent / "fixtures/window-v17-trace.json")
    assert old["engine_version"] == WINDOW_ENGINE_VERSION
    assert old["schema"] == "evidence-trace/8"
    assert old["retrieval"]["answer"]["status"] == "unsupported"
    assert replay(old)["verified"]
    assert make_trace(old["index"], old["retrieval"])["engine_version"] == WINDOW_ENGINE_VERSION
    new = search(old["index"], old["retrieval"]["query"])
    assert new["answer"]["status"] == "answered"
    assert new["answer"]["fields"]["reason"]["text"] == "道路积水"
    upgraded = make_trace(old["index"], new)
    assert upgraded["engine_version"] == ENGINE_VERSION
    assert upgraded["schema"] == "evidence-trace/10"
    assert upgraded["index"]["statistics"] == old["index"]["statistics"]
    assert replay(upgraded)["verified"]
