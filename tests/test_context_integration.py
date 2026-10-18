"""Cross-line evidence, independent ranking, provenance and versioned replay."""

import copy
from pathlib import Path

import pytest

from evidence import ANSWER_ENGINE_VERSION, ENGINE_VERSION
from evidence.dag_tokenization import default_lexicon
from evidence.ingest import IngestConfig
from evidence.query_expansion import resource_snapshot
from evidence.retrieval import build_index, digest, search
from evidence.tracing import deserialize, make_trace, read_record, replay, serialize

RESOURCE = resource_snapshot([], {"name": "context-integration-tests"})
NAME = "青禾社区雨水收集项目"
LEFT = NAME + "进入实施阶段。"
RIGHT = "该项目于2025年3月12日启动。"
QUESTION = NAME + "什么时候启动？"


def corpus(text, max_chars=400):
    return build_index(text.encode(), "多句短文.txt", "utf-8", [], IngestConfig(max_chars),
                       lexicon=default_lexicon(), expansion_resource=RESOURCE)


@pytest.mark.parametrize("separator", ["", "\n", "\n\n", "\r\n"])
def test_same_information_is_jointly_retrievable_under_different_paragraph_layouts(separator):
    text = LEFT + separator + RIGHT
    index = corpus(text, 16)
    result = search(index, QUESTION)
    answer = result["answer"]
    assert answer["status"] == "answered"
    assert answer["fields"]["time"]["text"] == "2025年3月12日"
    assert answer["evidence_ids"] == ["S000001", "S000002"]
    assert answer["bindings"][0]["checks"]["matching_reference_type"]
    assert index["context"]["statistics"]["N"] == 3
    pair = next(w for w in result["window_retrieval"]["candidates"] if len(w["sentence_ids"]) == 2)
    assert pair["source_text"] == text
    group = next(g for g in result["evidence_groups"]["groups"] if g["id"] in answer["evidence_group_ids"])
    assert [(s["sentence_id"], s["role"]) for s in group["source_spans"]] == [("S000001", "support"), ("S000002", "direct")]
    for field in answer["fields"].values():
        assert text[field["source_start"]:field["source_end"]] == field["text"]
    for span in group["source_spans"]:
        assert text[span["source_start"]:span["source_end"]] in (LEFT, RIGHT)


def test_window_statistics_are_independent_and_paragraph_index_is_still_auditable():
    index = corpus(LEFT + "\n" + RIGHT)
    result = search(index, QUESTION)
    assert index["statistics"]["N"] == 2
    assert result["window_retrieval"]["corpus"]["N"] == 3
    assert result["corpus"]["N"] == 2
    assert all(c["score"] == sum(t["contribution"] for t in c["term_details"])
               for c in result["window_retrieval"]["candidates"])


def test_overlapping_windows_do_not_repeat_same_evidence_or_add_bm25_scores():
    result = search(corpus(LEFT + "\n" + RIGHT + "\n储水箱容量为300升。"), QUESTION)
    groups = result["evidence_groups"]["groups"]
    assert len(groups) == 1
    assert groups[0]["sentence_ids"] == ["S000001", "S000002"]
    assert "S000003" not in groups[0]["sentence_ids"]
    assert groups[0]["coverage"] <= 1
    assert groups[0]["coverage"] == sum(f["contribution"] for f in groups[0]["features"])
    assert len(groups[0]["window_ids"]) >= 2


def test_single_sentence_answer_remains_minimal_when_neighbours_are_unrelated():
    result = search(corpus("雨水收集是将降雨储存后用于灌溉的方法。\n附近学校有20名教师。"), "雨水收集是什么？")
    assert result["answer"]["status"] == "answered"
    selected = next(g for g in result["evidence_groups"]["groups"] if g["id"] in result["answer"]["evidence_group_ids"])
    assert len(selected["sentence_ids"]) == 1


def test_unrelated_project_reference_does_not_grow_a_person_keyword_hit():
    result = search(corpus("李明参加读书活动。\n该项目于2025年启动。"), "李明")
    assert result["answer"]["status"] == "evidence_only"
    assert all("S000002" not in group["sentence_ids"] for group in result["evidence_groups"]["groups"])


def test_multiple_sentences_can_supply_keyword_evidence_without_asserting_a_binding():
    result = search(corpus(LEFT + "\n" + RIGHT), NAME + " 启动")
    assert result["answer"]["status"] == "evidence_only"
    assert not result["answer"]["fields"]
    assert any(len(g["sentence_ids"]) == 2 and g["context_sentence_ids"] for g in result["evidence_groups"]["groups"])
    assert not result["answer"]["bindings"]


@pytest.mark.parametrize("middle", ["\n项目概况\n", "\n第二章 其他\n", "\n1. 其他事项。\n"])
def test_titles_chapters_and_lists_do_not_become_anaphora_bridges(middle):
    result = search(corpus(LEFT + middle + RIGHT), QUESTION)
    assert result["answer"]["status"] != "answered"
    assert not result["answer"]["fields"]


def test_competing_project_and_general_pronoun_do_not_supply_a_false_answer():
    for text in (LEFT + "\n另一乙项目完成建设。\n" + RIGHT,
                 LEFT + "\n它于2025年3月12日启动。"):
        result = search(corpus(text), QUESTION)
        assert result["answer"]["status"] != "answered"
        assert not result["answer"]["fields"]


def test_conflicting_joint_dates_and_negated_joint_statement_block_a_unique_answer():
    for second in ("该项目于2025年4月20日启动。", "该项目并未于2025年4月20日启动。"):
        result = search(corpus(LEFT + "\n" + RIGHT + "\n" + LEFT + "\n" + second), QUESTION)
        assert result["answer"]["status"] == "ambiguous"
        assert not result["answer"]["fields"]
        assert result["answer"]["evidence_group_ids"]


def test_conflicting_disjoint_proofs_both_remain_linked_to_the_answer():
    text = LEFT + "\n" + RIGHT + "\n" + LEFT + "\n该项目于2025年4月20日启动。"
    result = search(corpus(text), QUESTION)
    assert result["answer"]["status"] == "ambiguous"
    ids = result["answer"]["evidence_group_ids"]
    assert len(ids) == 2
    assert [g["sentence_ids"] for g in result["evidence_groups"]["groups"] if g["id"] in ids] == [
        ["S000001", "S000002"], ["S000003", "S000004"]]


def test_returned_evidence_group_can_be_inspected_without_mutating_index_or_answer_proof():
    index = corpus(LEFT + "\n" + RIGHT)
    before = copy.deepcopy(index)
    result = search(index, QUESTION)
    binding = copy.deepcopy(result["answer"]["bindings"])
    group = result["evidence_groups"]["groups"][0]
    group["source_spans"][0]["location"]["start_line"] = 999
    group["binding"]["antecedent"]["text"] = "changed"
    assert result["answer"]["bindings"] == binding
    assert index == before


@pytest.fixture(scope="module")
def trace():
    index = corpus(LEFT + "\n" + RIGHT)
    return make_trace(index, search(index, QUESTION))


@pytest.mark.parametrize("target", ["window_df", "window_score", "group_role", "group_reason", "binding", "boundary"])
def test_resigned_window_group_or_joint_binding_tampering_fails_independent_replay(trace, target):
    altered = deserialize(serialize(trace))
    if target == "window_df":
        key = next(iter(altered["index"]["context"]["statistics"]["document_frequency"]))
        altered["index"]["context"]["statistics"]["document_frequency"][key] += 1
    elif target == "window_score":
        altered["retrieval"]["window_retrieval"]["candidates"][0]["score"] += 1
    elif target == "group_role":
        altered["retrieval"]["evidence_groups"]["groups"][0]["source_spans"][0]["role"] = "direct"
    elif target == "group_reason":
        altered["retrieval"]["evidence_groups"]["groups"][0]["inclusion_decisions"][0]["reason"] = "伪造关联"
    elif target == "binding":
        altered["retrieval"]["answer"]["bindings"][0]["reference"]["source_start"] += 1
    else:
        altered["index"]["context"]["boundaries"].append({"left_sentence_id": "S000001", "right_sentence_id": "S000002", "reasons": ["fake"]})
    altered["integrity_sha256"] = digest({k: v for k, v in altered.items() if k != "integrity_sha256"})
    with pytest.raises(ValueError, match="复算"):
        replay(altered)


def test_context_joint_groups_are_replayable(trace):
    report = replay(trace)
    assert report["verified"]
    assert any("窗口" in check for check in report["checks"])
    assert report["answer_status"] == "answered"
    assert report["evidence_group_ids"] == trace["retrieval"]["answer"]["evidence_group_ids"]


def test_frozen_16_trace_preserves_old_answers_and_migrates_to_17():
    old = read_record(Path(__file__).parent / "fixtures/answer-v16-trace.json")
    before = copy.deepcopy(old)
    assert old["engine_version"] == ANSWER_ENGINE_VERSION
    assert old["schema"] == "evidence-trace/7"
    assert replay(old)["verified"]
    assert "context" not in old["index"] and "window_retrieval" not in old["retrieval"]
    assert make_trace(old["index"], old["retrieval"])["engine_version"] == ANSWER_ENGINE_VERSION
    current = search(old["index"], old["retrieval"]["query"])
    upgraded = make_trace(old["index"], current)
    assert upgraded["engine_version"] == ENGINE_VERSION
    assert upgraded["schema"] == "evidence-trace/10"
    assert upgraded["index"]["statistics"] == old["index"]["statistics"]
    assert current["answer"]["fields"]["time"]["text"] == old["retrieval"]["answer"]["fields"]["time"]["text"]
    assert replay(upgraded)["verified"]
    assert old == before
