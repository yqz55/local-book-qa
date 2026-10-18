"""The original inspection workbench, available from Advanced Tools."""


def show_workbench():
    """Run with: python -m streamlit run app.py"""

    import json
    import os
    from pathlib import Path

    import streamlit as st

    from evidence import ENGINE_VERSION, ANSWER_ENGINE_VERSION
    from evidence.comparison import build_previous_index, comparison_manifest
    from evidence.ingest import IngestConfig
    from evidence.retrieval import BM25Config, build_index, configure_query_expansion, prepare_index_for_search, rebuild_with_aliases, rebuild_with_book_words, search
    from evidence.storage import Store
    from evidence.tokenization import Tokenizer
    from evidence.tracing import make_trace, read_record, rebuild_index, serialize
    from ui.styles import CSS
    from ui.comparison import show_comparison
    from ui.views import show_evidence, show_index, show_replay

    ROOT = Path(__file__).resolve().parents[1]
    STORE = Store(os.environ.get("NLP_DATA_DIR", str(ROOT / ".data")))
    WORDS = json.loads((ROOT / "data" / "dictionary.json").read_text(encoding="utf-8-sig"))



    @st.cache_resource(show_spinner=False)
    def load_index(path: str, modified: int, engine_version: str):
        return prepare_index_for_search(rebuild_index(read_record(path)))


    @st.cache_resource(show_spinner=False)
    def previous_index_for_comparison(index_id: str, previous_engine: str, _index):
        return build_previous_index(_index)


    def activate(index, path=None):
        index = prepare_index_for_search(index)
        if path is None or Path(path).name != f"{index['index_id']}.json.gz":
            target = STORE.root / "indexes" / f"{index['index_id']}.json.gz"
            path = target if target.is_file() else STORE.save_index(index)
        st.session_state.index = index
        st.session_state.active_path = str(path) if path else None
        st.session_state.result = None
        st.session_state.trace_bytes = None
        st.session_state.previous_index = None
        st.session_state.previous_result = None
        st.session_state.previous_trace_bytes = None
        st.session_state.comparison_manifest = None
        st.session_state.book_words_error = None
        st.session_state.book_aliases_error = None
        st.session_state.book_expansion_error = None
        for key in ("candidate_page", "previous_candidate_page", "paragraph_number", "term_page", "clean_page", "person_page", "person_detail", "dictionary_page"):
            st.session_state.pop(key, None)


    def run_query(query, config):
        index = st.session_state.index
        previous_index = previous_index_for_comparison(index["index_id"], ANSWER_ENGINE_VERSION, index)
        previous_result = search(previous_index, query, config, _engine_version=ANSWER_ENGINE_VERSION)
        result = search(index, query, config)
        manifest = comparison_manifest(index, result, previous_index, previous_result)
        previous_trace = make_trace(previous_index, previous_result)
        previous_data = serialize(previous_trace)
        trace = make_trace(index, result)
        data = serialize(trace)
        STORE.save_trace(previous_trace, previous_data)
        STORE.save_trace(trace, data)
        st.session_state.result, st.session_state.trace_bytes = result, data
        st.session_state.previous_index = previous_index
        st.session_state.previous_result = previous_result
        st.session_state.previous_trace_bytes = previous_data
        st.session_state.comparison_manifest = manifest


    def apply_book_words(expected_id, editor_key):
        """Callbacks finish before the next render, so widget IDs and tabs stay stable."""
        current = st.session_state.index
        try:
            if current["index_id"] != expected_id:
                raise ValueError("书籍版本已经变化，请重新编辑当前书籍的词条。")
            value = st.session_state[editor_key]
            words = Tokenizer([line.strip() for line in value.splitlines() if line.strip()]).words
            updated = rebuild_with_book_words(current, words)
            path = STORE.save_index(updated)
            activate(updated, path)
            st.session_state.book_select = str(path)
            st.session_state.rebuild_notice = "本书词条已应用：正文索引和查询词表已同步更新，原查询结果已清空，请重新检索。"
        except (ValueError, OSError, KeyError, TypeError) as exc:
            st.session_state.book_words_error = {"index_id": current["index_id"], "message": str(exc)}


    def apply_aliases(expected_id, editor_key):
        current = st.session_state.index
        try:
            if current["index_id"] != expected_id:
                raise ValueError("书籍版本已经变化，请重新编辑当前书籍的别名。")
            groups = [[name.strip() for name in line.replace("＝", "=").split("=")]
                      for line in st.session_state[editor_key].splitlines() if line.strip()]
            updated = rebuild_with_aliases(current, groups)
            path = STORE.save_index(updated)
            activate(updated, path)
            st.session_state.book_select = str(path)
            st.session_state.rebuild_notice = "人工关系校正已应用：姓名发现、正文索引及查询扩展已同步重建，请重新检索。"
        except (ValueError, OSError, KeyError, TypeError) as exc:
            st.session_state.book_aliases_error = {"index_id": current["index_id"], "message": str(exc)}


    def apply_query_expansion(expected_id, prefix):
        current = st.session_state.index
        try:
            if current["index_id"] != expected_id:
                raise ValueError("书籍版本已经变化，请重新编辑当前书籍的表达扩展。")
            groups = [[word.strip() for word in line.replace("＝", "=").split("=")]
                      for line in st.session_state[f"{prefix}_groups"].splitlines() if line.strip()]
            blocked = [line.strip() for line in st.session_state[f"{prefix}_blocked"].splitlines() if line.strip()]
            cfg = {**current["query_expansion"]["config"], "enabled": st.session_state[f"{prefix}_enabled"],
                   "weight": st.session_state[f"{prefix}_weight"], "max_alternatives": st.session_state[f"{prefix}_limit"]}
            updated = configure_query_expansion(current, config=cfg, groups=groups, blocked=blocked)
            path = STORE.save_index(updated)
            activate(updated, path)
            st.session_state.book_select = str(path)
            st.session_state.rebuild_notice = "表达扩展设置已保存，原查询结果已清空，请重新检索。"
        except (ValueError, OSError, KeyError, TypeError) as exc:
            st.session_state.book_expansion_error = {"index_id": current["index_id"], "message": str(exc)}


    if "pending_book_select" in st.session_state:
        st.session_state.book_select = st.session_state.pop("pending_book_select")

    if "index" in st.session_state and st.session_state.index["identity"]["engine_version"] != ENGINE_VERSION:
        with st.spinner("正在按当前分词与证据窗口规则重建全文索引…"):
            activate(st.session_state.index, st.session_state.get("active_path"))
            st.session_state.book_select = st.session_state.active_path
            st.session_state.rebuild_notice = "已按当前分词与证据窗口规则重建全文索引，请重新检索。"

    if "index" not in st.session_state:
        saved = STORE.list_indexes()
        try:
            if saved:
                path = Path(saved[0]["path"])
                activate(load_index(str(path), path.stat().st_mtime_ns, ENGINE_VERSION), path)
            else:
                index = build_index((ROOT / "data" / "demo.txt").read_bytes(), "演示·信息检索.txt", "utf-8", WORDS)
                activate(index, STORE.save_index(index))
                run_query("什么是信息检索", BM25Config())
        except (ValueError, OSError, KeyError, TypeError, EOFError) as exc:
            st.error(f"本地索引读取失败：{exc}。可从侧栏重新导入 TXT。")

    with st.sidebar:
        st.header("文本资料")
        upload = st.file_uploader("导入 TXT", type=["txt"], key="txt_upload", help="TXT 内容上限 30 MiB。")
        encoding = st.selectbox("文件编码", ["utf-8", "gb18030"],
                                format_func=lambda e: "UTF-8（严格）" if e == "utf-8" else "GB18030（严格）")
        with st.expander("切分与检索参数"):
            max_chars = st.number_input("每段最多字符数（建索引时应用）", 16, 4000, 400, step=16)
            k1 = st.number_input("k1（下次检索时应用）", 0.1, 10.0, st.session_state.get("retrieval_k1", 1.5), step=0.1, key="retrieval_k1")
            b = st.number_input("b（下次检索时应用）", 0.0, 1.0, st.session_state.get("retrieval_b", .75), step=0.05, key="retrieval_b")
            st.caption("展示最多 5 个检索窗口和 3 组证据。人物称呼按本书关系展开；通用词义扩展设置见“文本与索引”。")
            dictionary_upload = st.file_uploader("自定义词典（可选 JSON 汉字词数组）", type=["json"], key="dictionary_upload")
            st.caption("完整通用词典始终启用。上传词典作为补充；正文与查询使用同一份固定的整句分词配置。")
        if st.button("建立索引", type="primary", width="stretch", disabled=upload is None):
            try:
                words = json.loads(dictionary_upload.getvalue().decode("utf-8-sig")) if dictionary_upload else WORDS
                Tokenizer(words)
                with st.spinner("按明确规则建立本地索引…"):
                    index = build_index(upload.getvalue(), upload.name, encoding, words, IngestConfig(int(max_chars)))
                    path = STORE.save_index(index)
                    activate(index, path)
                    st.session_state.book_select = str(path)
                st.success("已建立索引。")
            except (ValueError, OSError, TypeError) as exc:
                st.error(str(exc))
        st.divider()
        saved = STORE.list_indexes()
        labels = {x["path"]: f"{x['filename']} · {x['index_id'][:8]}" for x in saved}
        if labels:
            paths = list(labels)
            current = st.session_state.get("active_path")
            selected = st.selectbox("已保存的文本", paths, index=paths.index(current) if current in paths else 0,
                                    format_func=lambda p: labels[p], key="book_select")
            if selected != current:
                try:
                    path = Path(selected)
                    activate(load_index(str(path), path.stat().st_mtime_ns, ENGINE_VERSION), path)
                    st.session_state.pending_book_select = st.session_state.active_path
                    st.rerun()
                except (ValueError, OSError, KeyError, TypeError, EOFError) as exc:
                    st.error(f"索引校验失败：{exc}")
        if st.button("使用演示文本", width="stretch"):
            try:
                index = build_index((ROOT / "data" / "demo.txt").read_bytes(), "演示·信息检索.txt", "utf-8", WORDS)
                activate(index, STORE.save_index(index))
                st.session_state.pop("book_select", None)
                st.session_state.search_query = "什么是信息检索"
                run_query("什么是信息检索", BM25Config(k1=k1, b=b))
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        if st.button("使用原因／方法示例", width="stretch"):
            try:
                index = build_index((ROOT / "data" / "discourse_demo.txt").read_bytes(), "原创示例·原因与方法.txt", "utf-8", WORDS)
                activate(index, STORE.save_index(index))
                st.session_state.pop("book_select", None)
                st.session_state.search_query = "青禾项目为什么会延期启动？"
                run_query("青禾项目为什么会延期启动？", BM25Config(k1=k1, b=b))
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        st.markdown('<div class="local-note">全部处理在本机完成<br>索引与查询历史保存在 .data 目录</div>', unsafe_allow_html=True)

    st.title("中文文本证据问答")
    st.markdown('<p class="subtitle">同一问题，对比上一版与当前版的答案、证据和计算过程。</p>', unsafe_allow_html=True)
    st.caption("当前版支持身份、属性、行为角色、地点、作用、描述及原文明示的原因和方法；其他问法提供带出处的原文参考回答。两版均可单独查看与导出复算。")
    notice = st.empty()  # Keep later widget paths stable when the transient notice disappears.
    if "rebuild_notice" in st.session_state:
        notice.success(st.session_state.pop("rebuild_notice"))
    if "index" not in st.session_state:
        st.info("请从侧栏导入 TXT，或使用演示文本。")
        st.stop()

    index = st.session_state.index
    stats = index["statistics"]
    st.markdown('<div class="stats">'
                f'<div class="stat"><small>原始检索段落数</small><strong>{stats["N"]:,}</strong></div>'
                f'<div class="stat"><small>词项数 · 词表大小</small><strong>{stats["vocabulary_size"]:,}</strong></div>'
                f'<div class="stat"><small>原始段落平均词项数</small><strong>{stats["avgdl"]:.2f}</strong></div>'
                f'<div class="stat"><small>文本版本</small><code>{index["source"]["text_sha256"][:12]}</code></div></div>', unsafe_allow_html=True)
    st.caption(f"当前资料：{index['source']['filename']} · 已保存索引 {index['index_id'][:12]}")
    st.session_state.setdefault("search_query", "什么是信息检索")
    with st.form("search_form"):
        question, submit = st.columns([6, 1.4], vertical_alignment="bottom")
        query = question.text_input("输入问题或关键词", max_chars=500, key="search_query")
        submitted = submit.form_submit_button("检索", type="primary", width="stretch")
    # Keep the tabs at one stable render position when query progress disappears.
    with st.container():
        if submitted:
            try:
                with st.spinner("分别运行上一版与当前版，保存两份复算记录…"):
                    run_query(query, BM25Config(k1=k1, b=b))
            except (ValueError, OSError) as exc:
                st.error(f"检索未完成：{exc}")

    # Chat runs only the current pipeline. Compute the frozen comparison when
    # the user explicitly enters this workbench, preserving the original trace.
    if st.session_state.result is not None and st.session_state.get("previous_result") is None:
        result = st.session_state.result
        previous_index = previous_index_for_comparison(index["index_id"], ANSWER_ENGINE_VERSION, index)
        previous_result = search(previous_index, result["query"], BM25Config(**result["config"]),
                                 _engine_version=ANSWER_ENGINE_VERSION)
        previous_trace = make_trace(previous_index, previous_result)
        previous_data = serialize(previous_trace)
        STORE.save_trace(previous_trace, previous_data)
        st.session_state.previous_index = previous_index
        st.session_state.previous_result = previous_result
        st.session_state.previous_trace_bytes = previous_data
        st.session_state.comparison_manifest = comparison_manifest(index, result, previous_index, previous_result)

    evidence, text, records = st.tabs(["检索证据", "文本与索引", "记录复算"])
    with evidence:
        show_comparison(index, st.session_state.result, st.session_state.trace_bytes,
                        st.session_state.get("previous_index"), st.session_state.get("previous_result"),
                        st.session_state.get("previous_trace_bytes"), st.session_state.get("comparison_manifest"))
    with text:
        show_index(index, apply_book_words, apply_aliases, apply_query_expansion)
    with records:
        show_replay(STORE)


if __name__ == "__main__":
    import streamlit as st
    from ui.styles import CSS
    st.set_page_config(page_title="中文文本证据问答 · 高级工具", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    show_workbench()
