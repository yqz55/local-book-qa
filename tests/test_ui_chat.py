"""Conversation isolation, real algorithm details and the new app entry flow."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import ChatInput
from streamlit.proto.WidgetStates_pb2 import WidgetState

from evidence.conversations import ConversationStore, export_conversation
from evidence.query_expansion import resource_snapshot
from evidence.retrieval import build_index, search
from evidence.storage import Store
from evidence.tracing import make_trace, replay, read_record
from ui.chat import answer_text, process_snapshot


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def chats(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_DATA_DIR", str(tmp_path))
    # Streamlit 1.51 AppTest still sends string_trigger_value for chat_input.
    # A file-enabled chat uses the chat_input_value wire format in real browsers.
    def chat_widget_state(widget):
        state = WidgetState(id=widget.id)
        if widget._value is not None:
            state.chat_input_value.data = widget._value
        return state
    monkeypatch.setattr(ChatInput, "_widget_state", property(chat_widget_state))
    return ConversationStore(Store(tmp_path))


def example_index(text="雨水收集是将降雨储存后用于灌溉的方法。\n示范项目于2025年启动。", filename="雨水收集.txt"):
    return build_index(text.encode(), filename, "utf-8", [], book_words=["雨水收集"],
                       expansion_resource=resource_snapshot([], {"name": "chat-ui-test"}))


def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()


def button(page, label):
    return next(item for item in page.button if item.label == label)


def ask(page, text):
    page.chat_input[0].set_value(text).run()
    assert not page.exception
    return page.session_state["result"]


def test_empty_app_requires_upload_and_has_no_automatic_demo_or_query(chats):
    page = app()
    assert not page.exception
    assert page.chat_input[0].disabled
    assert not page.chat_message
    assert not page.tabs
    assert not chats.list()
    assert not chats.store.list_indexes()
    assert not chats.store.list_traces()


def test_duplicate_uploads_are_separate_and_deleting_keeps_shared_index(chats):
    index = example_index()
    first, second = chats.create(index), chats.create(index)
    assert first["id"] != second["id"]
    assert first["index_id"] == second["index_id"]
    result = search(index, "雨水收集是什么？")
    trace = make_trace(index, result)
    trace_path = chats.store.save_trace(trace)
    chats.append(first["id"], trace, trace_path, process_snapshot(index, result))
    assert not chats.load(second["id"])["turns"]
    saved = chats.load(first["id"])
    exported = export_conversation(saved).decode("utf-8-sig")
    assert result["answer"]["text"] in exported and trace_path.stem.removesuffix(".json") in exported
    chats.rename(first["id"], "学习笔记")
    assert chats.load(first["id"])["title"] == "学习笔记"
    chats.clear(first["id"])
    assert not chats.load(first["id"])["turns"]
    assert replay(read_record(trace_path))["verified"]
    chats.delete(first["id"])
    assert len(chats.list()) == 1
    assert (chats.store.root / "indexes" / f"{index['index_id']}.json.gz").is_file()


def test_chat_app_keeps_multiple_independent_questions_and_collapses_process(chats):
    index = example_index()
    conversation = chats.create(index)
    page = app()
    assert not page.exception and not page.chat_input[0].disabled
    first = ask(page, "雨水收集是什么？")
    assert first == search(index, "雨水收集是什么？")
    assert page.session_state["previous_result"] is None
    assert len(page.chat_message) == 3
    second = ask(page, "示范项目什么时候启动？")
    assert second == search(index, "示范项目什么时候启动？")
    assert len(page.chat_message) == 5
    saved = chats.load(conversation["id"])
    assert len(saved["turns"]) == 2
    assert saved["turns"][0]["query"] == first["query"]
    assert saved["turns"][1]["query"] == second["query"]
    assert len([item for item in page.expander if item.label == "查看算法过程"]) == 2
    assert all(not item.proto.expanded for item in page.expander if item.label == "查看算法过程")
    assert not page.tabs
    assert saved["turns"][0]["process"]["window_count"] == first["window_retrieval"]["candidate_count"]
    assert any("<mark>" in quote["html"] for quote in saved["turns"][0]["process"]["quotes"])
    reopened = app()
    assert not reopened.exception
    assert len(reopened.chat_message) == 5
    assert reopened.session_state["result"] == second
    assert len(chats.store.list_traces()) == 2


def test_switch_rename_clear_delete_and_undo_are_isolated(chats):
    first = chats.create(example_index())
    second = chats.create(example_index("观测项目于2024年启动。", "观测记录.txt"))
    page = app()
    assert page.session_state["active_conversation_id"] == second["id"]
    ask(page, "观测项目什么时候启动？")
    button(page, first["title"]).click().run()
    assert not page.exception
    assert page.session_state["active_conversation_id"] == first["id"]
    assert len(page.chat_message) == 1
    ask(page, "雨水收集是什么？")
    page.text_input[0].set_value("雨水学习笔记")
    button(page, "重命名").click().run()
    assert not page.exception and chats.load(first["id"])["title"] == "雨水学习笔记"
    button(page, "清空当前问答").click().run()
    assert not page.exception and len(page.chat_message) == 1
    assert len(chats.load(second["id"])["turns"]) == 1
    button(page, "删除对话").click().run()
    assert not page.exception and len(chats.list()) == 1
    button(page, "撤销删除").click().run()
    assert not page.exception and len(chats.list()) == 2
    assert page.session_state["active_conversation_id"] == first["id"]
    button(page, "新建对话").click().run()
    assert not page.exception and page.chat_input[0].disabled
    assert page.session_state["active_conversation_id"] is None


def test_advanced_tools_use_selected_trace_and_return_without_losing_chat(chats):
    conversation = chats.create(example_index())
    page = app()
    result = ask(page, "雨水收集是什么？")
    data = page.session_state["trace_bytes"]
    button(page, "在高级工具中查看完整记录").click().run()
    assert not page.exception
    assert page.session_state["workbench_mode"]
    assert len(page.tabs) == 3
    assert page.session_state["result"] == result
    assert page.session_state["trace_bytes"] == data
    assert page.session_state["previous_result"] is not None
    button(page, "返回对话").click().run()
    assert not page.exception and not page.tabs
    assert len(page.chat_message) == 3
    assert len(chats.load(conversation["id"])["turns"]) == 1


def test_chat_escapes_imported_markup_and_no_content_has_useful_answer(chats):
    index = example_index("记录中包含 <script>alert(1)</script>。")
    chats.create(index)
    page = app()
    ask(page, "<b>记录</b>")
    assert any("&lt;b&gt;记录&lt;/b&gt;" in item.value for item in page.markdown)
    assert all("<script>" not in quote["html"] for quote in process_snapshot(index, page.session_state["result"])["quotes"])
    assert "具体对象" in answer_text({"status": "no_content", "text": ""})
    assert answer_text({"status": "evidence_only", "mode": "reference", "text": "原句"}).startswith("原文参考：")


def test_advanced_dictionary_changes_keep_old_answers_and_other_conversations(chats):
    original = example_index()
    other = chats.create(original)
    current = chats.create(original)
    page = app()
    ask(page, "雨水收集是什么？")
    old_turn = chats.load(current["id"])["turns"][0]
    button(page, "高级工具").click().run()
    assert not page.exception
    editor = next(item for item in page.text_area if item.label == "本书专用词条（每行一个）")
    editor.set_value("雨水收集\n示范项目")
    button(page, "应用词条并重建索引").click().run()
    assert not page.exception
    next(item for item in page.number_input if item.key == "retrieval_b").set_value(.5).run()
    button(page, "返回对话").click().run()
    assert not page.exception
    assert chats.load(current["id"])["index_id"] != original["index_id"]
    assert chats.load(current["id"])["turns"][0] == old_turn
    assert chats.load(other["id"])["index_id"] == original["index_id"]
    result = ask(page, "示范项目什么时候启动？")
    assert result["config"]["b"] == .5
    assert len(chats.load(current["id"])["turns"]) == 2
    assert replay(read_record(chats.trace_path(old_turn["trace_id"])))["verified"]
    button(page, "高级工具").click().run()
    assert not page.exception
    assert next(item for item in page.number_input if item.key == "retrieval_b").value == .5
    button(page, "用当前文本新建对话").click().run()
    assert not page.exception and len(page.chat_message) == 1
    assert page.session_state["active_conversation_id"] not in (current["id"], other["id"])
    assert len(chats.load(current["id"])["turns"]) == 2
    assert page.session_state["chat_config"]["b"] == .5


@pytest.mark.parametrize("bad_id", ["../indexes/anything", "", "x" * 32])
def test_conversation_identifiers_cannot_escape_store(chats, bad_id):
    with pytest.raises(ValueError):
        chats.load(bad_id)
