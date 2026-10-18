"""Cross-domain answer coverage and adversarial source/role preservation."""

from copy import deepcopy
from pathlib import Path

import pytest

from evidence import DISCOURSE_ENGINE_VERSION, ENGINE_VERSION, schema_version
from evidence.query_expansion import resource_snapshot
from evidence.retrieval import build_index, digest, prepare_index_for_search, search
from evidence.tracing import make_trace, read_record, replay


EMPTY = resource_snapshot([], {"name": "general-answer-acceptance"})


def ask(text, query, **kwargs):
    index = build_index(text.encode("utf-8"), "通用问答验收.txt", "utf-8", [], expansion_resource=EMPTY, **kwargs)
    return index, search(index, query)


def check_spans(index, answer):
    def walk(value):
        if isinstance(value, dict):
            if "source_start" in value:
                assert index["source"]["text"][value["source_start"]:value["source_end"]] == value["text"]
            else:
                for item in value.values():
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(answer["fields"])
    walk(answer["bindings"])


@pytest.mark.parametrize("text,query,value", [
    ("弗莱明发现了青霉素。", "谁发现了青霉素？", "弗莱明"),
    ("张衡发明了地动仪。", "张衡发明了什么？", "地动仪"),
    ("鲁迅创作了小说集。", "谁创作了小说集？", "鲁迅"),
    ("研究所发布了观测报告。", "研究所发布了什么？", "观测报告"),
    ("陈晨修复了望远镜。", "谁修复了望远镜？", "陈晨"),
    ("李明击败了王强。", "李明击败了谁？", "王强"),
    ("班长是李明。", "谁是班长？", "李明"),
    ("李明是班长。", "班长是谁？", "李明"),
    ("孔子是教育家。", "孔子是谁？", "教育家"),
    ("盒子的颜色是蓝色。", "盒子的颜色是什么？", "蓝色"),
    ("盒子的颜色是蓝色。", "盒子是什么颜色？", "蓝色"),
    ("桥梁的材料为钢材。", "桥梁的材料是什么？", "钢材"),
    ("展览的主题是海洋。", "展览的主题是什么？", "海洋"),
    ("研究所位于北京。", "研究所在哪里？", "北京"),
    ("会议在上海举办。", "会议在哪里举办？", "上海"),
    ("弗莱明发现了青霉素。", "青霉素是谁发现的？", "弗莱明"),
    ("青霉素由弗莱明发现。", "青霉素被谁发现？", "弗莱明"),
    ("张衡发明了地动仪。", "地动仪是由谁发明的？", "张衡"),
    ("张衡发明了地动仪。", "请问书中谁发明了地动仪？", "张衡"),
])
def test_source_bound_short_answers_across_entities_and_predicates(text, query, value):
    index, result = ask(text, query)
    answer = result["answer"]
    assert answer["status"] == "answered", answer
    assert answer["fields"].get("value", {}).get("text") == value, answer
    assert answer["evidence_ids"]
    assert answer["evidence_group_ids"]
    check_spans(index, answer)


@pytest.mark.parametrize("query", ["作者认为谁是最可爱的人？", "作者认为最可爱的人是谁？"])
def test_author_opinion_and_same_sentence_topic_have_separate_proofs(query):
    text = "谁是我们最可爱的人呢？我们的战士，我感到他们是最可爱的人。"
    index, result = ask(text, query)
    answer = result["answer"]
    assert answer["status"] == "answered", answer
    assert answer["statement_kind"] == "opinion"
    assert answer["fields"]["value"]["text"] == "我们的战士"
    assert answer["fields"]["attribution"]["text"] == "我感到"
    assert answer["bindings"][0]["reference"]["text"] == "他们"
    check_spans(index, answer)


@pytest.mark.parametrize("text,query", [
    ("这个连虽然伤亡很大，他们却打死了三百多敌人，更重要的，他们使得我们部队的主力赶上来，聚歼了敌人。", "这个连重要的贡献是什么？"),
    ("水库的主要作用是调节水量，保障下游供水。", "水库的主要作用是什么？"),
    ("研究所组织了观测活动。", "研究所做了什么？"),
])
def test_complete_source_excerpt_can_answer_beyond_narrow_fields(text, query):
    index, result = ask(text, query)
    answer = result["answer"]
    assert answer["status"] == "answered", answer
    assert answer["mode"] == "excerpt"
    assert answer["fields"]["description"]["text"] == text
    check_spans(index, answer)


@pytest.mark.parametrize("text,query", [
    ("甲击败了乙。", "乙击败了谁？"),
    ("甲要求乙发现青霉素。", "谁发现青霉素？"),
    ("甲认为乙发现青霉素。", "谁发现青霉素？"),
    ("据说弗莱明发现了青霉素。", "谁发现了青霉素？"),
    ("弗莱明可能发现了青霉素。", "青霉素是谁发现的？"),
    ("林清认为战士是最可爱的人。", "作者认为谁是最可爱的人？"),
    ("他们是最可爱的人。", "作者认为谁是最可爱的人？"),
    ("李明批评他们，我感到他们是最可爱的人。", "作者认为谁是最可爱的人？"),
    ("研究所没有在北京。", "研究所在哪里？"),
    ("乙盒子的颜色是蓝色。", "甲盒子的颜色是什么？"),
    ("弗莱明发现了青霉素", "谁发现了青霉素？"),
    ("研究所位于北京。", "研究所为什么位于北京？"),
])
def test_generic_fallback_cannot_bypass_role_state_and_boundary_guards(text, query):
    _, result = ask(text, query)
    answer = result["answer"]
    assert answer["status"] != "answered", answer
    assert not answer["fields"]
    assert answer["text"]


@pytest.mark.parametrize("query", [
    "请概括研究所的主要贡献。", "研究所的成果有哪些？", "怎样评价研究所的工作？",
    "研究所是否完成观测？", "研究所什么时候成立，负责人是谁？", "研究所与学校有什么不同？",
    "研究所的未来会怎样？", "研究所的研究对社会有什么启示？",
])
def test_open_questions_receive_real_source_reference_answers(query):
    source = "研究所组织了观测活动。研究所发布了观测报告。"
    index, result = ask(source, query)
    answer = result["answer"]
    assert answer["mode"] == "reference", answer
    assert answer["status"] == "evidence_only"
    assert not answer["fields"]
    assert answer["references"]
    assert "原文参考" in answer["text"]
    for row in answer["references"]:
        assert row["text"] == source[row["source_start"]:row["source_end"]]
        assert row["text"] in answer["text"]


def test_no_content_and_no_evidence_have_explicit_responses():
    _, result = ask("研究所发布了观测报告。", "为什么？")
    assert result["answer"]["status"] == "no_content"
    _, result = ask("研究所发布了观测报告。", "火星天气如何？")
    assert result["answer"]["status"] == "insufficient_evidence"
    assert result["answer"]["mode"] == "no_evidence"
    assert not result["answer"]["fields"]


@pytest.mark.parametrize("source,query", [
    ("甲发现了样品。乙发现了样品。", "谁发现了样品？"),
    ("甲发现了样品。甲没有发现样品。", "谁发现了样品？"),
])
def test_competing_or_negated_answer_does_not_choose_first(source, query):
    _, result = ask(source, query)
    assert result["answer"]["status"] == "ambiguous", result["answer"]
    assert not result["answer"]["fields"]
    assert len(result["answer"]["evidence_ids"]) >= 2


def test_raw_whitespace_mapping_and_general_snapshot_replay():
    index, result = ask("\ufeff　盒子的颜色是\t蓝色。　\r\n", "盒子的颜色是什么？")
    assert result["answer"]["status"] == "answered"
    check_spans(index, result["answer"])
    trace = make_trace(index, result)
    assert trace["engine_version"] == ENGINE_VERSION
    assert trace["schema"] == f"evidence-trace/{schema_version(ENGINE_VERSION)}"
    assert replay(trace)["verified"]


@pytest.mark.parametrize("part", ["question_plan", "general_rules", "answer_field", "reference_text"])
def test_resigned_general_record_changes_are_detected(part):
    query = "盒子的颜色是什么？" if part != "reference_text" else "盒子有什么启示？"
    index, result = ask("盒子的颜色是蓝色。", query)
    trace = deepcopy(make_trace(index, result))
    if part == "question_plan":
        trace["retrieval"]["question"]["answer_plan"]["subject"] = "别的对象"
    elif part == "general_rules":
        trace["retrieval"]["general_answer_policy"]["rules"]["QA101"] = "伪造解释"
    elif part == "answer_field":
        trace["retrieval"]["answer"]["fields"]["value"]["text"] = "红色"
    else:
        trace["retrieval"]["answer"]["references"][0]["text"] = "不存在的原文"
    trace["integrity_sha256"] = digest({k: v for k, v in trace.items() if k != "integrity_sha256"})
    with pytest.raises(ValueError, match="复算"):
        replay(trace)


def test_frozen_18_replays_and_upgrades_without_rewriting_history():
    path = Path(__file__).parent / "fixtures" / "discourse-v18-trace.json"
    trace = read_record(path)
    original = deepcopy(trace)
    assert trace["engine_version"] == DISCOURSE_ENGINE_VERSION
    assert replay(trace)["verified"]
    index = prepare_index_for_search(trace["index"])
    result = search(index, "作者认为谁是最可爱的人？")
    assert result["answer"]["status"] == "answered", result["answer"]
    assert replay(make_trace(index, result))["verified"]
    assert trace == original


def test_state_description_keeps_a_complete_answer_instead_of_assuming_method():
    text = "烈士们的遗体，保留着各种各样的姿势，有抱住敌人腰的，有抱住敌人头的。"
    index, result = ask(text, "烈士们的遗体保持着怎样的姿势？")
    assert result["answer"]["status"] == "answered", result["answer"]
    assert result["answer"]["mode"] == "excerpt"
    assert result["answer"]["fields"]["description"]["text"] == text
    check_spans(index, result["answer"])


def test_describing_another_subject_does_not_fill_a_state_slot():
    _, result = ask("遗体已经被搬走，展品保留着各种各样的姿势。", "遗体保持着怎样的姿势？")
    assert result["answer"]["status"] != "answered"


def test_static_definition_cannot_answer_what_subject_did():
    _, result = ask("研究所是成立了多年的建筑。", "研究所做了什么？")
    assert result["answer"]["status"] != "answered"


def test_aspect_particle_matching_cannot_delete_characters_inside_names():
    _, result = ask("然修复了望远镜。", "了然修复了什么？", book_words=["了然"])
    assert result["answer"]["status"] != "answered"
    _, result = ask("张衡发明地动仪。", "张衡发明了什么？")
    assert result["answer"]["status"] == "answered"
    assert result["answer"]["fields"]["value"]["text"] == "地动仪"


@pytest.mark.parametrize("boundary,allowed", [("\n", True), ("\n第二章\n", False), ("\n- ", False)])
def test_adjacent_quotation_is_explicit_reference_context_with_boundaries(boundary, allowed):
    source = "营长回忆了战斗。" + boundary + "他说：“我为战士们骄傲。”"
    index, result = ask(source, "营长为什么哭？")
    answer = result["answer"]
    assert answer["status"] != "answered"
    assert not answer["fields"]
    assert bool(answer.get("reference_context_ids")) is allowed
    if allowed:
        assert "我为战士们骄傲" in answer["text"]
        assert not set(answer["reference_context_ids"]).intersection(answer["evidence_ids"])
        assert replay(make_trace(index, result))["verified"]
