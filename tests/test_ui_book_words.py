from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_book_editor_keeps_widget_state_after_rebuild_and_rejected_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui" / "workbench.py"), default_timeout=30).run()
    assert not app.exception
    previous_id = app.session_state["index"]["index_id"]
    assert app.session_state["result"] is not None

    def submit(value):
        app.text_area[0].set_value(value)
        next(b for b in app.button if b.label == "应用词条并重建索引").click().run()
        assert not app.exception

    submit("光合作用")
    updated_id = app.session_state["index"]["index_id"]
    assert updated_id != previous_id
    assert app.session_state["result"] is None
    assert app.session_state["trace_bytes"] is None

    # The prior success notice expires on this render. Widget paths must stay stable.
    submit("abc")
    assert app.session_state["index"]["index_id"] == updated_id
    assert len(app.error) == 1
    assert app.session_state["book_words_error"]["index_id"] == updated_id

    submit("")
    assert app.session_state["index"]["index_id"] == previous_id
    assert app.session_state["book_words_error"] is None
