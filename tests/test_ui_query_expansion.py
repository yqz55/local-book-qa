from pathlib import Path

from streamlit.testing.v1 import AppTest

from evidence.retrieval import build_index
from evidence.storage import Store
from evidence.tracing import deserialize, replay


def test_book_expansion_settings_search_and_invalid_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_DATA_DIR", str(tmp_path))
    index = build_index("甲把书赠与乙。\n乙把书送给甲。\n甲送别乙。".encode(), "赠送演示.txt", "utf-8", [])
    Store(tmp_path).save_index(index)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui" / "workbench.py"), default_timeout=45).run()
    assert not app.exception
    next(t for t in app.text_input if t.label == "输入问题或关键词").set_value("赠送")
    next(b for b in app.button if b.label == "检索").click().run()
    assert not app.exception
    assert len(app.session_state["result"]["top_ids"]) == 2
    assert replay(deserialize(app.session_state["trace_bytes"]))["verified"]
    stats = app.session_state["index"]["statistics"]
    next(c for c in app.checkbox if c.label == "启用通用词义扩展").uncheck()
    next(b for b in app.button if b.label == "应用表达扩展设置").click().run()
    assert not app.exception
    assert app.session_state["result"] is None
    assert app.session_state["index"]["statistics"] == stats
    next(b for b in app.button if b.label == "检索").click().run()
    assert app.session_state["result"]["status"] == "no_matches"
    next(c for c in app.checkbox if c.label == "启用通用词义扩展").check()
    next(t for t in app.text_area if t.label == "本书停用扩展表达（每行一个）").set_value("送给")
    next(b for b in app.button if b.label == "应用表达扩展设置").click().run()
    next(b for b in app.button if b.label == "检索").click().run()
    assert not app.exception
    assert app.session_state["result"]["top_ids"] == ["P000001"]
    previous_id = app.session_state["index"]["index_id"]
    next(t for t in app.text_area if t.label == "本书表达校正（每行一组，用 = 分隔）").set_value("赠送 = 没有")
    next(b for b in app.button if b.label == "应用表达扩展设置").click().run()
    assert not app.exception
    assert app.session_state["index"]["index_id"] == previous_id
    assert app.session_state["book_expansion_error"]
