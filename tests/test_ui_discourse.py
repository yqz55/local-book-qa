"""Cause and method answers remain source-bound in every comparison view."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
from streamlit.testing.v1 import AppTest

from evidence import ANSWER_ENGINE_VERSION, ENGINE_VERSION
from evidence.retrieval import build_index
from evidence.storage import Store
from evidence.tracing import deserialize
from ui.answers import discourse_source_fields, sentence_quote


def _source_fixture(kind):
    if kind == "reason":
        raw = "青禾项目暂停。\r\n原因是设备故障。"
        values = {"target": "青禾项目", "effect": "暂停", "reason": "设备故障"}
    else:
        raw = "雨水收集操作方法。\r\n先清洗<滤网>，再安装滤芯。"
        values = {"target": "雨水收集", "method": "先清洗<滤网>，再安装滤芯"}
    texts = raw.split("\r\n")
    records = []
    for number, text in enumerate(texts, 1):
        start = raw.index(text)
        records.append({"id": f"S{number}", "sentence_id": f"S{number}", "text": text,
                        "source_start": start, "source_end": start + len(text),
                        "location": {"start_line": number, "start_column": 1}})
    fields = {}
    for name, value in values.items():
        start = raw.index(value)
        fields[name] = {"text": value, "source_start": start, "source_end": start + len(value),
                        "sentence_id": "S2" if start >= records[1]["source_start"] else "S1"}
    answer = {"status": "answered", "text": "原文表述：" + values[kind], "fields": fields,
              "statement_kind": "procedure" if kind == "method" else "fact", "evidence_ids": ["S1", "S2"],
              "evidence_group_ids": ["G1"]}
    group = {"id": "G1", "sentence_ids": ["S1", "S2"], "supporting_sentence_ids": ["S1"],
             "direct_sentence_ids": ["S2"], "context_sentence_ids": [],
             "source_spans": [{**record, "role": "support" if record["id"] == "S1" else "direct",
                               "reason": "原文对象关联" if record["id"] == "S1" else "包含原文字段"}
                              for record in records]}
    return ({"source": {"text": raw}, "sentences": {"sentences": records}},
            {"answer": answer, "question": {"type": kind, "target": values["target"], "keywords": [values["target"]]},
             "sentence_evidence": {"candidates": records}, "evidence_groups": {"groups": [group], "top_ids": ["G1"]}})


@pytest.mark.parametrize("kind", ["reason", "method"])
def test_discourse_panel_displays_literal_source_fields_and_matching_highlights(kind):
    index, result = _source_fixture(kind)
    # The displayed value is always the raw-source slice, even if a cached label
    # were stale. The citation and highlighted range therefore remain identical.
    result["answer"]["fields"][kind]["text"] = "不能替代原文的缓存文本"
    rows = discourse_source_fields(index, result["answer"])
    code = ("import json\nfrom ui.answers import show_answer\n"
            f"index = json.loads({json.dumps(index)!r})\n"
            f"result = json.loads({json.dumps(result)!r})\nshow_answer(index, result)\n")
    app = AppTest.from_string(code, default_timeout=30).run()
    assert not app.exception
    assert any(widget.value == "**答案字段与原文出处**" for widget in app.markdown)
    for row in rows:
        assert row["原文内容"] in [widget.value for widget in app.text]
        assert any(row["字段"] in widget.value and row["来源句"] in widget.value
                   and row["原文字符"] in widget.value for widget in app.caption)
    assert not any("缓存文本" in widget.value for widget in app.text)
    assert any("关联证据 · S1" in widget.value for widget in app.caption)
    assert any("直接证据 · S2" in widget.value for widget in app.caption)
    if kind == "reason":
        assert any("原因关系取自原文明确表述" in widget.value for widget in app.caption)
        assert any("<mark>设备故障</mark>" in widget.value for widget in app.markdown)
    else:
        assert any("不代表这些操作已经执行" in widget.value for widget in app.caption)
        assert any("<mark>先清洗&lt;滤网&gt;，再安装滤芯</mark>" in widget.value for widget in app.markdown)
        assert not any("<滤网>" in widget.value for widget in app.markdown)


def test_discourse_source_panel_requires_an_answer_and_valid_source_ranges():
    index, result = _source_fixture("reason")
    answer = result["answer"]
    assert discourse_source_fields(index, {**answer, "status": "evidence_only"}) == []
    assert discourse_source_fields(index, {**answer, "status": "unsupported"}) == []
    assert discourse_source_fields(index, {**answer, "fields": {"reason": {"source_start": -1, "source_end": 1}}}) == []
    assert "<mark>设备故障</mark>" in sentence_quote(index, result["sentence_evidence"]["candidates"][1], answer["fields"])


def test_original_discourse_example_button_runs_real_comparison_and_keeps_existing_demo(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui" / "workbench.py"), default_timeout=90).run()
    assert not app.exception
    assert any(widget.label == "使用演示文本" for widget in app.button)
    next(widget for widget in app.button if widget.label == "使用原因／方法示例").click().run()
    assert not app.exception
    assert app.session_state["index"]["source"]["filename"] == "原创示例·原因与方法.txt"
    assert next(widget for widget in app.text_input if widget.label == "输入问题或关键词").value == "青禾项目为什么会延期启动？"
    assert app.session_state["result"]["query"] == "青禾项目为什么会延期启动？"
    assert app.session_state["result"]["answer"]["status"] == "answered"
    assert app.session_state["previous_result"]["answer"]["status"] == "unsupported"
    assert all(app.session_state["comparison_manifest"]["checks"].values())
    assert len(Store(tmp_path).list_traces()) == 4
    next(widget for widget in app.text_input if widget.label == "输入问题或关键词").set_value("青禾小组如何完成土壤调查？")
    next(widget for widget in app.button if widget.label == "检索").click().run()
    assert not app.exception
    assert app.session_state["result"]["answer"]["status"] == "answered"
    assert "method" in app.session_state["result"]["answer"]["fields"]


@pytest.mark.parametrize("article,query,field,statement_kind", [
    ("因为施工道路积水，所以青禾项目延期启动。", "青禾项目为什么会延期启动？", "reason", "fact"),
    ("青禾小组通过分区采样完成土壤调查。", "青禾小组如何完成土壤调查？", "method", "fact"),
    ("要完成土壤调查，可以通过分区采样。", "怎样才能完成土壤调查？", "method", "procedure"),
])
def test_natural_discourse_comparison_keeps_previous_unanswered_and_both_exports_replayable(
        article, query, field, statement_kind, tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_DATA_DIR", str(tmp_path))
    Store(tmp_path).save_index(build_index(article.encode("utf-8"), "原因方法短文.txt", "utf-8", []))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui" / "workbench.py"), default_timeout=90).run()
    assert not app.exception
    next(widget for widget in app.text_input if widget.label == "输入问题或关键词").set_value(query)
    next(widget for widget in app.button if widget.label == "检索").click().run()
    assert not app.exception
    answer = app.session_state["result"]["answer"]
    assert answer["status"] == "answered"
    assert answer["statement_kind"] == statement_kind
    assert field in answer["fields"]
    # The frozen 1.6 parser has no 怎样才能 expression: the question tokenizes as
    # one word and falls back to keyword evidence instead of its unsupported
    # branch. Both are non-answers, and neither may carry a reason/method field.
    previous = app.session_state["previous_result"]["answer"]
    assert previous["status"] in {"unsupported", "evidence_only", "insufficient_evidence"}
    assert not {"reason", "method"}.intersection(previous["fields"])
    assert all(app.session_state["comparison_manifest"]["checks"].values())
    assert any(widget.value == "**答案字段与原文出处**" for widget in app.markdown)
    assert any("直接证据 ·" in widget.value for widget in app.caption)
    assert any("<mark>" in widget.value for widget in app.markdown)
    if statement_kind == "procedure":
        assert any("不代表这些操作已经执行" in widget.value for widget in app.caption)
        assert any("原文语气" in widget.value for widget in app.caption)
        assert "可以" in [widget.value for widget in app.text]
    assert [tab.label for tab in app.tabs] == ["检索证据", "文本与索引", "记录复算"]
    summary = app.dataframe[0].value
    assert list(summary["版本"]) == [f"上一版 · {ANSWER_ENGINE_VERSION.rsplit('/', 1)[-1]}",
                                   f"当前版 · {ENGINE_VERSION.rsplit('/', 1)[-1]}"]
    for key, version in (("previous_trace_bytes", ANSWER_ENGINE_VERSION), ("trace_bytes", ENGINE_VERSION)):
        data = app.session_state[key]
        assert deserialize(data)["engine_version"] == version
        path = tmp_path / f"{key}.json.gz"
        path.write_bytes(data)
        process = subprocess.run([sys.executable, "-m", "evidence.cli", "replay", str(path)],
                                 cwd=Path(__file__).resolve().parents[1], capture_output=True,
                                 encoding="utf-8", timeout=90)
        assert process.returncode == 0, process.stderr
        assert json.loads(process.stdout)["verified"]
    for view in ("当前版", "上一版", "并排对比"):
        next(widget for widget in app.radio if widget.label == "结果展示方式").set_value(view).run()
        assert not app.exception
        assert app.session_state["result"]["answer"] == answer
