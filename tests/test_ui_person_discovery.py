from pathlib import Path

from streamlit.testing.v1 import AppTest

from evidence.retrieval import build_index
from evidence.storage import Store
from evidence.tracing import deserialize, replay


def widget(items, label):
    return next(w for w in items if w.label == label)


def test_automatic_names_query_evidence_and_manual_correction_are_separate(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_DATA_DIR", str(tmp_path))
    text = "赵清岚说道：好。\n清岚点头。\n张若溪说道：好。\n李若溪起身。\n若溪点头。"
    index = build_index(text.encode(), "自动人物.txt", "utf-8", [])
    Store(tmp_path).save_index(index)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "ui" / "workbench.py"), default_timeout=30).run()
    assert not app.exception
    assert widget(app.text_area, "本书别名（每行一组，用 = 分隔）").value == ""
    widget(app.text_input, "筛选姓名或称呼").set_value("清岚").run()
    assert not app.exception
    assert "赵清岚" in widget(app.selectbox, "查看候选判断与原文位置").options
    widget(app.text_input, "输入问题或关键词").set_value("赵清岚")
    widget(app.button, "检索").click().run()
    assert not app.exception
    result = app.session_state["result"]
    assert {t["term"] for t in result["query_terms"]} == {"清岚", "赵清岚"}
    assert result["query_expansions"][0]["origin"] == "automatic"
    assert replay(deserialize(app.session_state["trace_bytes"]))["verified"]
    widget(app.text_input, "输入问题或关键词").set_value("若溪")
    widget(app.button, "检索").click().run()
    assert not app.exception
    assert any("未自动合并这些人物" in w.value for w in app.warning)
    assert app.session_state["result"]["query_ambiguities"]


def test_filtering_resets_out_of_range_person_page(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_DATA_DIR", str(tmp_path))
    surnames = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳"
    text = "\n".join(f"{s}清岚说道：好。" for s in surnames)
    Store(tmp_path).save_index(build_index(text.encode(), "多候选.txt", "utf-8", []))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "ui" / "workbench.py"), default_timeout=30).run()
    assert not app.exception
    widget(app.number_input, "姓名候选页码（每页 50 个）").set_value(2).run()
    assert not app.exception
    widget(app.text_input, "筛选姓名或称呼").set_value("赵清岚").run()
    assert not app.exception
    assert widget(app.number_input, "姓名候选页码（每页 50 个）").value == 1
