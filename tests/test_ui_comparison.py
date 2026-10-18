from pathlib import Path

from streamlit.testing.v1 import AppTest

from evidence import ANSWER_ENGINE_VERSION, ENGINE_VERSION
from evidence.retrieval import build_index
from evidence.storage import Store
from evidence.tracing import deserialize, replay


def test_comparison_modes_preserve_both_results_exports_and_clear_together(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_DATA_DIR", str(tmp_path))
    text = "青禾项目进入实施阶段。\n该项目于2025年3月12日启动。"
    Store(tmp_path).save_index(build_index(text.encode(), "两版对比.txt", "utf-8", []))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui" / "workbench.py"), default_timeout=90).run()
    assert not app.exception
    next(t for t in app.text_input if t.label == "输入问题或关键词").set_value("青禾项目什么时候启动？")
    next(b for b in app.button if b.label == "检索").click().run()
    assert not app.exception
    new, old = app.session_state["result"], app.session_state["previous_result"]
    assert new["answer"]["status"] == "answered"
    assert old["answer"]["status"] == "evidence_only"
    assert all(app.session_state["comparison_manifest"]["checks"].values())
    previous_trace = deserialize(app.session_state["previous_trace_bytes"])
    current_trace = deserialize(app.session_state["trace_bytes"])
    assert previous_trace["engine_version"] == ANSWER_ENGINE_VERSION
    assert current_trace["engine_version"] == ENGINE_VERSION
    assert replay(previous_trace)["verified"] and replay(current_trace)["verified"]
    assert [tab.label for tab in app.tabs] == ["检索证据", "文本与索引", "记录复算"]
    assert next(r for r in app.radio if r.label == "结果展示方式").value == "并排对比"
    labels = [item.value for item in app.markdown]
    assert any("上一版 · 段落召回与单句答案" in label for label in labels)
    assert any("当前版 · 窗口召回与多句答案" in label for label in labels)
    history_count = len(Store(tmp_path).list_traces())
    for view in ("上一版", "当前版", "并排对比"):
        next(r for r in app.radio if r.label == "结果展示方式").set_value(view).run()
        assert not app.exception
        assert app.session_state["result"] == new and app.session_state["previous_result"] == old
        assert len(Store(tmp_path).list_traces()) == history_count
    next(t for t in app.text_area if t.label == "本书停用扩展表达（每行一个）").set_value("赠送")
    next(b for b in app.button if b.label == "应用表达扩展设置").click().run()
    assert not app.exception
    assert app.session_state["result"] is None and app.session_state["previous_result"] is None
    assert app.session_state["trace_bytes"] is None and app.session_state["previous_trace_bytes"] is None
