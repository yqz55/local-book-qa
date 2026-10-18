"""Chat presentation and compact, factual explanations of the real search run."""

from copy import deepcopy
import html

import streamlit as st

from .answers import ANSWER_STATES, QUESTION_TYPES, sentence_quote
from .context_evidence import group_sentence_spans, visible_groups
from .views import query_source_label


def process_snapshot(index, result):
    """Keep the history small without fabricating or recomputing an explanation."""
    answer = result["answer"]
    windows = result.get("window_retrieval", result)
    selected = windows.get("candidates", [])[:5]
    quotes, seen = [], set()
    for group in visible_groups(result):
        for sentence in group_sentence_spans(index, result, group):
            if sentence["sentence_id"] in seen:
                continue
            seen.add(sentence["sentence_id"])
            quotes.append({"sentence_id": sentence["sentence_id"], "role": sentence["role"],
                           "reason": sentence.get("reason", ""), "chapter": sentence.get("chapter", ""),
                           "source_start": sentence["source_start"], "source_end": sentence["source_end"],
                           "html": sentence_quote(index, sentence, answer.get("fields", {}))})
    # Sentence-only records and no-match runs need no invented evidence groups.
    if not quotes:
        for sentence in result.get("sentence_evidence", {}).get("candidates", [])[:3]:
            quotes.append({"sentence_id": sentence["sentence_id"], "role": "direct" if
                           sentence["sentence_id"] in answer.get("evidence_ids", []) else "context",
                           "reason": "答案引用句" if sentence["sentence_id"] in answer.get("evidence_ids", []) else "候选原句",
                           "chapter": sentence.get("chapter", ""), "source_start": sentence["source_start"],
                           "source_end": sentence["source_end"],
                           "html": sentence_quote(index, sentence, answer.get("fields", {}))})
    return deepcopy({
        "question": result.get("question", {}),
        "tokens": [{"term": token["term"], "surface": token["surface"], "rule": token["rule"]}
                   for token in result.get("query_tokens", [])],
        "terms": [{"term": term["term"], "weight": term.get("weight", 1),
                   "source": query_source_label(term)} for term in result.get("query_terms", [])],
        "window_count": windows.get("candidate_count", 0), "corpus": windows.get("corpus", {}),
        "config": result.get("config", {}), "paragraph_count": result.get("candidate_count", 0),
        "windows": [{key: window.get(key) for key in
                     ("window_id", "paragraph_id", "score", "sentence_ids", "term_details", "dl", "length_normalization")}
                    for window in selected],
        "quotes": quotes, "status": answer.get("status"), "mode": answer.get("mode"),
        "reason": answer.get("reason", ""), "evidence_ids": answer.get("evidence_ids", []),
        "rules": answer.get("rule_ids", answer.get("rules", [])),
        "decisions": answer.get("decisions", []),
    })


def answer_text(answer):
    text = answer.get("text", "").strip()
    if answer.get("mode") == "reference" and text and not text.startswith("原文参考"):
        return "原文参考：\n" + text
    if text:
        return text
    fallbacks = {
        "no_content": "请在问题中写明具体对象或关键词，我会在当前文档中查找答案。",
        "ambiguous": "文档中有多个候选或相互冲突的证据，暂时无法确定唯一答案。请补充具体对象或条件。",
        "insufficient_evidence": "当前文档中未找到足够证据来回答这个问题。",
        "unsupported": "当前问法暂时无法从文档中得到可核对的答案。",
    }
    return fallbacks.get(answer.get("status"), "当前文档中未找到可核对的答案。")


def show_process(process):
    question = process.get("question", {})
    st.markdown("**1. 理解问题并分词**")
    st.caption("清洗问题，使用与正文相同的词典和分词规则，识别提问类型与检索内容。")
    st.text("问题类型：" + QUESTION_TYPES.get(question.get("type"), question.get("type", "关键词检索")))
    if process["tokens"]:
        st.dataframe([{"实际文字": token["surface"], "词项": token["term"], "规则": token["rule"]}
                      for token in process["tokens"]], hide_index=True, width="stretch")
    keywords = question.get("keywords", [])
    if keywords:
        st.text("实际检索关键词：" + "、".join(str(word) for word in keywords))
    if question.get("constraints"):
        st.json({"问题约束": question["constraints"]}, expanded=False)

    st.markdown("**2. 确定检索词与扩展**")
    st.caption("根据本书姓名关系和词义资源筛选扩展；下表是本次实际使用的词项及权重。")
    if process["terms"]:
        st.dataframe([{"检索词项": term["term"], "来源": term["source"], "权重": term["weight"]}
                      for term in process["terms"]], hide_index=True, width="stretch")
    else:
        st.text("本次没有可用于检索的内容词。")

    st.markdown("**3. 召回候选并计算排序**")
    st.text(f"召回 {process['window_count']} 个句子窗口；原始段落诊断召回 {process['paragraph_count']} 段。")
    corpus, config = process["corpus"], process["config"]
    st.caption(f"窗口 N = {corpus.get('N', 0)} · avgdl = {corpus.get('avgdl', 0):.4f} · "
               f"k1 = {config.get('k1', 1.5)} · b = {config.get('b', .75)}。分数用于排序，不表示正确概率。")
    if process["windows"]:
        st.dataframe([{"排名": rank, "候选": window.get("window_id") or window.get("paragraph_id"),
                       "句子": "、".join(window.get("sentence_ids") or []), "BM25": window["score"]}
                      for rank, window in enumerate(process["windows"], 1)], hide_index=True, width="stretch")
        with st.expander("查看前 5 个候选的逐词计分"):
            st.code("IDF = ln(1 + (N − DF + 0.5) / (DF + 0.5))\n"
                    "L = 1 − b + b × dl / avgdl\n"
                    "贡献 = 权重 × IDF × TF × (k1 + 1) / (TF + k1 × L)\n"
                    "总分 = 实际贡献之和；同组只取最大贡献", language=None)
            for window in process["windows"]:
                st.caption(str(window.get("window_id") or window.get("paragraph_id")))
                details = window.get("term_details") or []
                if details:
                    st.dataframe([{"词项": term["term"], "TF": term.get("tf", 0),
                                   "DF": term.get("df", 0), "IDF": term.get("idf", 0),
                                   "权重": term.get("weight", 1), "实际贡献": term.get("contribution", 0)}
                                  for term in details], hide_index=True, width="stretch")

    st.markdown("**4. 核对原文证据与关系**")
    st.caption("区分直接证据、对象关联和补充上下文，再检查问题约束及答案字段。")
    labels = {"direct": "直接证据", "support": "关联证据", "context": "补充上下文"}
    for quote in process["quotes"]:
        usage = "答案引用" if quote["sentence_id"] in process.get("evidence_ids", []) else "候选参考"
        st.caption(f"{usage} · {labels.get(quote['role'], quote['role'])} · {quote['sentence_id']} · "
                   f"{quote.get('chapter') or '未标章节'} · 原文字符 {quote['source_start'] + 1}–{quote['source_end']}")
        # Escaped from the original source when the trace was created.
        st.markdown(f'<div class="evidence-text">{quote["html"]}</div>', unsafe_allow_html=True)
        if quote.get("reason"):
            st.caption("加入原因：" + quote["reason"])
    if not process["quotes"]:
        st.text("本次没有找到相关的原文证据。")

    st.markdown("**5. 形成答案或说明证据不足**")
    st.text(ANSWER_STATES.get(process.get("status"), "原文参考回答"))
    if process.get("reason"):
        st.text(process["reason"])
    if process.get("evidence_ids"):
        st.caption("实际引用句：" + "、".join(process["evidence_ids"]))
    if process.get("rules"):
        st.caption("采用的规则：" + "、".join(str(rule) for rule in process["rules"]))
    if process.get("decisions"):
        with st.expander("查看抽取、排除及规则判断"):
            st.json(process["decisions"], expanded=False)


def show_ready(index):
    with st.chat_message("assistant", avatar=":material/description:"):
        st.markdown("**文件已准备好**")
        st.markdown("已完成文本清洗、分词与检索索引，使用通用词典与本书词表。")
        st.caption(f"{index['source']['filename']} · {index['statistics']['N']:,} 个检索段落 · "
                   f"{index['statistics']['vocabulary_size']:,} 个词项")


def show_turn(turn, on_inspect):
    with st.chat_message("user"):
        st.markdown(f'<div class="user-question">{html.escape(turn["query"])}</div>', unsafe_allow_html=True)
    with st.chat_message("assistant", avatar=":material/menu_book:"):
        # Render escaped text directly: Markdown lists plus pre-wrap otherwise
        # preserve generated whitespace and inflate gaps between references.
        paragraphs = "".join(f"<p>{html.escape(paragraph)}</p>"
                             for paragraph in answer_text(turn["answer"]).split("\n\n") if paragraph.strip())
        st.html(f'<div class="answer-text">{paragraphs}</div>')
        with st.expander("查看算法过程"):
            show_process(turn["process"])
            st.button("在高级工具中查看完整记录", key=f"inspect_{turn['id']}",
                      icon=":material/open_in_new:", on_click=on_inspect, args=(turn["trace_id"],))
