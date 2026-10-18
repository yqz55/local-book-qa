"""Exercise alias editor callbacks through the rendered Streamlit forms."""

import copy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evidence.tracing import read_record


ROOT = Path(__file__).resolve().parents[1]
ALIAS_LABEL = "本书别名（每行一组，用 = 分隔）"
WORDS_LABEL = "本书专用词条（每行一个）"


def widget(elements, label):
    matches = [element for element in elements if element.label == label]
    assert len(matches) == 1, f"Expected one widget labelled {label!r}"
    return matches[0]


def submit_aliases(app, value):
    widget(app.text_area, ALIAS_LABEL).set_value(value)
    widget(app.button, "应用别名并重建索引").click().run()
    assert not app.exception


def submit_words(app, value):
    widget(app.text_area, WORDS_LABEL).set_value(value)
    widget(app.button, "应用词条并重建索引").click().run()
    assert not app.exception


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_DATA_DIR", str(tmp_path))
    page = AppTest.from_file(str(ROOT / "ui" / "workbench.py"), default_timeout=30).run()
    assert not page.exception
    return page


def test_manual_corrections_start_empty_and_valid_edit_rebuilds_and_clears_query(app):
    before = app.session_state["index"]
    previous_id = before["index_id"]
    assert before["aliases"]["manual_groups"] == []
    assert widget(app.text_area, ALIAS_LABEL).value == ""
    assert app.session_state["result"] is not None
    assert app.session_state["trace_bytes"] is not None

    # Both ASCII and full-width equal signs are accepted by the form parser.
    submit_aliases(app, "黛玉 ＝ 林黛玉 = 林姑娘")
    updated = app.session_state["index"]
    group = sorted(["黛玉", "林黛玉", "林姑娘"])
    assert updated["index_id"] != previous_id
    assert updated["source"] == before["source"]
    assert updated["aliases"]["groups"] == [group]
    assert updated["dictionary"]["book_words"] == before["dictionary"]["book_words"]
    assert app.session_state["result"] is None
    assert app.session_state["trace_bytes"] is None
    assert app.session_state["book_aliases_error"] is None
    assert not app.error
    assert read_record(app.session_state["active_path"]) == updated
    assert app.session_state["book_select"] == app.session_state["active_path"]

    # The transient success notice expires; the new form still has the right state.
    app.run()
    assert not app.exception
    editor = widget(app.text_area, ALIAS_LABEL)
    assert editor.key == f"book_aliases_{updated['index_id']}"
    assert editor.value == " = ".join(group)


def test_invalid_overlap_keeps_current_index_query_and_attempted_editor_text(app):
    submit_aliases(app, "林黛玉 = 黛玉 = 林姑娘")
    widget(app.text_input, "输入问题或关键词").set_value("信息检索")
    widget(app.button, "检索").click().run()
    assert not app.exception
    index_before = copy.deepcopy(app.session_state["index"])
    query_before = copy.deepcopy(app.session_state["result"])
    trace_before = app.session_state["trace_bytes"]
    active_path = app.session_state["active_path"]
    assert query_before is not None

    attempted = "林黛玉 = 黛玉\n黛玉 = 潇湘妃子"
    submit_aliases(app, attempted)
    assert app.session_state["index"] == index_before
    assert app.session_state["result"] == query_before
    assert app.session_state["trace_bytes"] == trace_before
    assert app.session_state["active_path"] == active_path
    error = app.session_state["book_aliases_error"]
    assert error["index_id"] == index_before["index_id"]
    assert "同一名称不能属于多个别名组" in error["message"]
    assert len(app.error) == 1
    assert app.error[0].value == error["message"]
    assert widget(app.text_area, ALIAS_LABEL).value == attempted

    app.run()
    assert not app.exception
    assert len(app.error) == 1
    assert widget(app.text_area, ALIAS_LABEL).value == attempted
    submit_aliases(app, "林黛玉 = 黛玉 = 潇湘妃子")
    assert app.session_state["book_aliases_error"] is None
    assert not app.error
    assert app.session_state["result"] is None


def test_word_and_alias_edits_preserve_each_other_and_empty_groups_survive_reload(app):
    submit_aliases(app, "林黛玉 = 黛玉 = 林姑娘")
    configured_groups = copy.deepcopy(app.session_state["index"]["aliases"]["groups"])
    submit_words(app, "光合作用\n专用术语")
    expected_words = sorted(["光合作用", "专用术语"])
    assert app.session_state["index"]["dictionary"]["book_words"] == expected_words
    assert app.session_state["index"]["aliases"]["groups"] == configured_groups

    submit_aliases(app, "林黛玉 = 黛玉 = 颦儿")
    assert app.session_state["index"]["dictionary"]["book_words"] == expected_words
    assert app.session_state["index"]["aliases"]["groups"] == [sorted(["林黛玉", "黛玉", "颦儿"])]
    submit_aliases(app, "")
    assert app.session_state["index"]["aliases"]["groups"] == []
    assert app.session_state["index"]["dictionary"]["book_words"] == expected_words
    assert widget(app.text_area, ALIAS_LABEL).value == ""

    # A later dictionary rebuild must preserve the explicit choice of no aliases.
    submit_words(app, "光合作用")
    empty_index = copy.deepcopy(app.session_state["index"])
    assert empty_index["aliases"]["groups"] == []
    assert empty_index["dictionary"]["book_words"] == ["光合作用"]
    assert read_record(app.session_state["active_path"]) == empty_index
    app.run()
    assert not app.exception
    assert app.session_state["index"] == empty_index
    assert widget(app.text_area, ALIAS_LABEL).value == ""

    # Start a fresh session against the same store, exercising saved-index loading.
    reopened = AppTest.from_file(str(ROOT / "ui" / "workbench.py"), default_timeout=30).run()
    assert not reopened.exception
    assert reopened.session_state["index"] == empty_index
    assert widget(reopened.text_area, ALIAS_LABEL).value == ""
    assert reopened.session_state["index"]["dictionary"]["book_words"] == ["光合作用"]
    assert not reopened.error
