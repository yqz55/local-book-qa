"""Run with: python -m streamlit run app.py"""

import html
import json
import os
from pathlib import Path

import streamlit as st

from evidence import ENGINE_VERSION
from evidence.conversations import ConversationStore, export_conversation
from evidence.retrieval import BM25Config, build_index, prepare_index_for_search, search
from evidence.storage import Store
from evidence.tracing import make_trace, read_record, rebuild_index, serialize
from ui.chat import process_snapshot, show_ready, show_turn
from ui.source import show_original
from ui.styles import CSS


ROOT = Path(__file__).resolve().parent
STORE = Store(os.environ.get("NLP_DATA_DIR", str(ROOT / ".data")))
CHATS = ConversationStore(STORE)
WORDS = json.loads((ROOT / "data" / "dictionary.json").read_text(encoding="utf-8-sig"))
READ_ERRORS = (ValueError, OSError, KeyError, TypeError, EOFError)

st.set_page_config(page_title="文本问答", page_icon=":material/description:", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_index(path, modified, version):
    return prepare_index_for_search(rebuild_index(read_record(path)))


def reset_results():
    for key in ("result", "trace_bytes", "previous_index", "previous_result", "previous_trace_bytes", "comparison_manifest"):
        st.session_state[key] = None
    for key in ("candidate_page", "previous_candidate_page", "paragraph_number", "term_page", "clean_page",
                "person_page", "person_detail", "dictionary_page"):
        st.session_state.pop(key, None)


def use_index(index):
    st.session_state.index = index
    path = STORE.root / "indexes" / f"{index['index_id']}.json.gz"
    if not path.is_file():
        STORE.save_index(index)
    st.session_state.active_path = str(path)
    st.session_state.book_select = str(path)
    reset_results()


def open_chat(conversation_id):
    st.session_state.active_conversation_id = conversation_id
    st.session_state.workbench_mode = False
    st.session_state.pop("loaded_conversation_id", None)


def new_chat():
    st.session_state.active_conversation_id = None
    st.session_state.workbench_mode = False
    st.session_state.upload_generation = st.session_state.get("upload_generation", 0) + 1
    for key in ("index", "active_path", "loaded_conversation_id"):
        st.session_state.pop(key, None)
    reset_results()


def start_conversation(index):
    conversation = CHATS.create(index)
    use_index(index)
    open_chat(conversation["id"])
    st.session_state.loaded_conversation_id = conversation["id"]
    return conversation


def restore_conversation(conversation):
    path = STORE.root / "indexes" / f"{conversation['index_id']}.json.gz"
    index = load_index(str(path), path.stat().st_mtime_ns, ENGINE_VERSION)
    if index["index_id"] != conversation["index_id"]:
        STORE.save_index(index)
        conversation = CHATS.change_index(conversation["id"], index)
    use_index(index)
    turns = conversation["turns"]
    if turns and turns[-1]["index_id"] == index["index_id"]:
        try:
            trace_path = CHATS.trace_path(turns[-1]["trace_id"])
            trace = read_record(trace_path)
            st.session_state.result = trace["retrieval"]
            st.session_state.trace_bytes = trace_path.read_bytes()
        except READ_ERRORS:
            pass  # Saved answers remain usable if a diagnostic was removed.
    st.session_state.loaded_conversation_id = conversation["id"]
    return conversation


def upload_conversation(upload):
    data = upload.getvalue()
    if len(data) > 30 * 1024 * 1024:
        raise ValueError("TXT 文件不能超过 30 MiB。")
    if not upload.name.lower().endswith(".txt"):
        raise ValueError("请选择 TXT 文件。")
    try:
        data.decode("utf-8-sig", errors="strict")
        encoding = "utf-8"
    except UnicodeDecodeError:
        try:
            data.decode("gb18030", errors="strict")
            encoding = "gb18030"
        except UnicodeDecodeError as exc:
            raise ValueError("无法识别文件编码，请将 TXT 保存为 UTF-8 或 GB18030 后上传。") from exc
    with st.spinner("正在清洗文本、分词并建立检索索引…"):
        index = build_index(data, upload.name, encoding, WORDS)
        if not index["statistics"]["N"]:
            raise ValueError("文件中没有可检索的文字，请上传包含正文的 TXT。")
        return start_conversation(index)


def ask(query, conversation):
    if not query.strip():
        return
    with st.spinner("正在检索文档并核对答案…"):
        index = st.session_state.index
        result = search(index, query, BM25Config(**st.session_state.get("chat_config", {})))
        trace = make_trace(index, result)
        data = serialize(trace)
        path = STORE.save_trace(trace, data)
        CHATS.append(conversation["id"], trace, path, process_snapshot(index, result))
        reset_results()
        st.session_state.result, st.session_state.trace_bytes = result, data


def open_tools(trace_id=None):
    if trace_id:
        try:
            path = CHATS.trace_path(trace_id)
            trace = read_record(path)
            use_index(trace["index"])
            st.session_state.result = trace["retrieval"]
            st.session_state.trace_bytes = path.read_bytes()
        except READ_ERRORS as exc:
            st.session_state.notice = f"完整记录读取失败：{exc}"
            return
    st.session_state.workbench_origin_index = st.session_state.get("index", {}).get("index_id")
    config = st.session_state.get("chat_config", {})
    st.session_state.retrieval_k1 = config.get("k1", 1.5)
    st.session_state.retrieval_b = config.get("b", .75)
    st.session_state.workbench_mode = True


def back_to_chat():
    current_id = st.session_state.get("active_conversation_id")
    index = st.session_state.get("index")
    if current_id and index:
        conversation = CHATS.load(current_id)
        if index["index_id"] != st.session_state.get("workbench_origin_index"):
            old_path = STORE.root / "indexes" / f"{conversation['index_id']}.json.gz"
            old = read_record(old_path)
            if index["source"]["text_sha256"] == old["source"]["text_sha256"]:
                CHATS.change_index(current_id, index)
            else:
                start_conversation(index)
    st.session_state.chat_config = {"k1": st.session_state.get("retrieval_k1", 1.5),
                                     "b": st.session_state.get("retrieval_b", .75)}
    st.session_state.workbench_mode = False
    st.session_state.pop("loaded_conversation_id", None)


def chat_from_workbench():
    index = st.session_state.get("index")
    if index:
        st.session_state.chat_config = {"k1": st.session_state.get("retrieval_k1", 1.5),
                                         "b": st.session_state.get("retrieval_b", .75)}
        start_conversation(index)


def rename_chat(conversation_id, widget_key):
    try:
        CHATS.rename(conversation_id, st.session_state[widget_key])
    except READ_ERRORS as exc:
        st.session_state.notice = str(exc)


def remove_chat(conversation_id):
    st.session_state.deleted_conversation = CHATS.load(conversation_id)
    CHATS.delete(conversation_id)
    remaining = CHATS.list()
    new_chat()
    if remaining:
        open_chat(remaining[0]["id"])


def undo_delete():
    conversation = st.session_state.pop("deleted_conversation", None)
    if conversation:
        CHATS.save(conversation)
        open_chat(conversation["id"])


def clear_chat(conversation_id):
    CHATS.clear(conversation_id)
    reset_results()


if "active_conversation_id" not in st.session_state:
    saved = CHATS.list()
    st.session_state.active_conversation_id = saved[0]["id"] if saved else None
st.session_state.setdefault("workbench_mode", False)

if st.session_state.workbench_mode:
    back, create = st.columns([1, 2])
    back.button("返回对话", icon=":material/arrow_back:", on_click=back_to_chat)
    create.button("用当前文本新建对话", icon=":material/add_comment:", on_click=chat_from_workbench)
    from ui.workbench import show_workbench
    show_workbench()
    st.stop()

conversation = None
if st.session_state.active_conversation_id:
    try:
        conversation = CHATS.load(st.session_state.active_conversation_id)
        if st.session_state.get("loaded_conversation_id") != conversation["id"]:
            with st.spinner("正在打开文档…"):
                conversation = restore_conversation(conversation)
    except READ_ERRORS as exc:
        st.error(f"对话读取失败：{exc}。可新建对话重新上传文件。")
        conversation = None

with st.sidebar:
    st.markdown('<div class="app-brand"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
                '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
                '<path d="M14 2v6h6M8 13h8M8 17h6"/></svg>文本问答</div>', unsafe_allow_html=True)
    st.button("新建对话", icon=":material/add:", width="stretch", on_click=new_chat, key="new_chat")
    st.markdown('<div class="history-label">对话记录</div>', unsafe_allow_html=True)
    with st.container(key="conversation_list"):
        saved = CHATS.list()
        if not saved:
            st.caption("上传一份 TXT，开始第一个对话")
        for saved_chat in saved:
            st.button(saved_chat["title"], icon=":material/description:", width="stretch",
                      type="primary" if conversation and saved_chat["id"] == conversation["id"] else "tertiary",
                      key=f"chat_{saved_chat['id']}", help=saved_chat["filename"],
                      on_click=open_chat, args=(saved_chat["id"],))
    with st.container(key="sidebar_footer"):
        if st.session_state.get("deleted_conversation"):
            st.button("撤销删除", icon=":material/undo:", on_click=undo_delete, width="stretch")
        st.button("高级工具", icon=":material/settings:", width="stretch", type="tertiary", on_click=open_tools)
        st.markdown('<div class="local-note">本地处理 · 自动保存</div>', unsafe_allow_html=True)

if st.session_state.get("notice"):
    st.warning(st.session_state.pop("notice"))

if conversation:
    header, actions = st.columns([12, 1], vertical_alignment="center")
    with header:
        st.markdown(f'<h1 class="document-title">{html.escape(conversation["filename"])}</h1>'
                    '<p class="document-subtitle">每次提问独立检索当前文档</p>', unsafe_allow_html=True)
    with actions:
        with st.popover("", icon=":material/more_vert:", help="对话操作"):
            rename_key = f"rename_{conversation['id']}"
            with st.form(f"rename_form_{conversation['id']}"):
                st.text_input("对话名称", value=conversation["title"], key=rename_key, max_chars=100)
                st.form_submit_button("重命名", on_click=rename_chat, args=(conversation["id"], rename_key), width="stretch")
            st.download_button("导出聊天记录", export_conversation(conversation),
                               file_name=f"chat-{conversation['id'][:8]}.txt", mime="text/plain", icon=":material/download:",
                               on_click="ignore", width="stretch")
            st.button("清空当前问答", icon=":material/delete_sweep:", width="stretch", on_click=clear_chat,
                      args=(conversation["id"],), disabled=not conversation["turns"])
            st.button("删除对话", icon=":material/delete:", width="stretch", on_click=remove_chat, args=(conversation["id"],))
    st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)
    show_original(st.session_state.index)
    with st.container(key="chat_history"):
        show_ready(st.session_state.index)
        for turn in conversation["turns"]:
            show_turn(turn, open_tools)
else:
    with st.container(key="empty_state"):
        st.markdown('<div class="empty-document" aria-hidden="true">'
                    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" '
                    'stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
                    '<path d="M14 2v6h6M8 13h8M8 17h6"/></svg></div>'
                    '<h1>从一份文档，开始提问</h1><p>上传 TXT，建立索引后即可提问</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("上传 TXT，开始对话", type=["txt"],
                                    key=f"initial_upload_{st.session_state.get('upload_generation', 0)}",
                                    label_visibility="collapsed", help="支持 UTF-8、GB18030，单个 TXT 不超过 30 MiB。")
        st.caption("支持 UTF-8、GB18030 · 单个文件不超过 30 MiB")
        if uploaded:
            try:
                upload_conversation(uploaded)
                st.rerun()
            except READ_ERRORS as exc:
                st.error(str(exc))

prompt_key = f"prompt_{conversation['id']}" if conversation else "prompt_empty"
submitted = st.chat_input("针对当前文档提问…" if conversation else "请先上传 TXT 文档…",
                           key=prompt_key, max_chars=500, accept_file=True, file_type=["txt"], disabled=conversation is None)
if submitted:
    try:
        text = submitted if isinstance(submitted, str) else submitted.text
        files = [] if isinstance(submitted, str) else submitted.files
        if files:
            conversation = upload_conversation(files[0])
        if text.strip() and conversation:
            ask(text, conversation)
        st.rerun()
    except READ_ERRORS as exc:
        st.error(f"未能完成本次操作：{exc}")
