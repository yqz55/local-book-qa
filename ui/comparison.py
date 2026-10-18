"""Two real pipeline outputs, presented together under one query."""

import streamlit as st

from .answers import ANSWER_STATES, show_answer
from .context_evidence import show_window_retrieval
from .views import show_evidence


def show_comparison(index, result, trace_bytes, previous_index, previous_result, previous_trace_bytes, manifest):
    st.subheader("版本对比")
    view = st.radio("结果展示方式", ["并排对比", "当前版", "上一版"], horizontal=True, key="comparison_view")
    st.caption("同一文本、问题、分词词典、人工校正和 BM25 参数分别运行两版；两版的 BM25 分数使用不同检索单位，请分别查看计算依据。")
    if result is None:
        st.info("输入问题并点击“检索”，同时查看两版答案和证据。")
        return
    if previous_result is None:
        st.info("上一版结果尚未生成，请点击“检索”运行两版对比。")
        return
    rows = []
    for title, version_index, item in (("上一版", previous_index, previous_result), ("当前版", index, result)):
        version = version_index["identity"]["engine_version"].rsplit("/", 1)[-1]
        label = f"{title} · {version}"
        retrieval = item.get("window_retrieval", item)
        state_label = ANSWER_STATES.get(item["answer"]["status"], item["answer"]["status"])
        if item["answer"].get("mode") == "reference":
            state_label = "原文参考回答"
        elif item["answer"].get("mode") == "excerpt":
            state_label = "已形成原句答案"
        rows.append({"版本": label, "答案状态": state_label,
                     "检索单位": "句子窗口" if "window_retrieval" in item else "原始段落",
                     "召回候选数": retrieval["candidate_count"], "答案引用句数": len(item["answer"]["evidence_ids"])})
    st.dataframe(rows, hide_index=True, width="stretch", key="comparison_summary")
    st.caption(f"本次对比问题：{result['query']} · k1 = {result['config']['k1']}，b = {result['config']['b']}")

    def previous():
        st.markdown("### 上一版 · 段落召回与单句答案")
        st.caption(f"算法版本：{previous_index['identity']['engine_version'].rsplit('/', 1)[-1]} · 冻结规则")
        st.download_button("导出上一版复算记录", previous_trace_bytes,
                           file_name=f"trace-previous-{previous_index['index_id'][:12]}.json.gz",
                           mime="application/gzip", key="comparison_previous_trace", width="stretch")
        show_answer(previous_index, previous_result)
        with st.expander("上一版段落检索、位置与计算明细"):
            show_evidence(previous_index, previous_result, previous_trace_bytes, key_prefix="previous_")

    def current():
        st.markdown("### 当前版 · 窗口召回与多句答案")
        st.caption(f"算法版本：{index['identity']['engine_version'].rsplit('/', 1)[-1]} · 当前规则")
        st.download_button("导出当前版复算记录", trace_bytes,
                           file_name=f"trace-current-{index['index_id'][:12]}.json.gz",
                           mime="application/gzip", key="comparison_current_trace", width="stretch")
        show_answer(index, result)
        show_window_retrieval(index, result)
        with st.expander("原始段落检索诊断与记录导出"):
            st.caption("以下保留原始段落 BM25 供核对。当前版规则答案使用独立计算的句子窗口召回结果。")
            show_evidence(index, result, trace_bytes)

    if view == "并排对比":
        left, right = st.columns(2, gap="large")
        with left, st.container(border=True):
            previous()
        with right, st.container(border=True):
            current()
    elif view == "上一版":
        previous()
    else:
        current()
    with st.expander("对比输入与版本核对"):
        st.json(manifest, expanded=False)
