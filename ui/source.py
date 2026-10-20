"""Read the uploaded document without applying index cleaning or Markdown."""

import base64
import html

import streamlit as st


PAGE_CHARS = 12_000


def show_original(index):
    with st.expander("查看原文", icon=":material/description:"):
        _show_original_content(index)


@st.fragment
def _show_original_content(index):
    source = index["source"]
    text = source["text"]
    pages = max(1, (len(text) + PAGE_CHARS - 1) // PAGE_CHARS)
    st.caption(f"{source['filename']} · 共 {len(text):,} 字符 · 保留原始空白与换行")
    page = 1
    if pages > 1:
        page = st.number_input("原文页码", min_value=1, max_value=pages, value=1, step=1,
                               key=f"original_page_{index['index_id']}")
    start = (page - 1) * PAGE_CHARS
    end = min(start + PAGE_CHARS, len(text))
    st.caption(f"第 {page} / {pages} 页 · 原文字符 {start + 1:,}–{end:,}")
    with st.container(height=400):
        st.markdown(f'<div class="original-text">{html.escape(text[start:end])}</div>',
                    unsafe_allow_html=True)
    st.download_button("下载原始 TXT", base64.b64decode(source["bytes_base64"]),
                       file_name=source["filename"], mime="application/octet-stream",
                       key=f"original_download_{index['index_id']}", on_click="ignore",
                       icon=":material/download:")
