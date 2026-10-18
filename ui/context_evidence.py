"""Render window retrieval and multi-sentence evidence without joining quotes."""

from copy import deepcopy

import streamlit as st


ROLE_LABELS = {"direct": "直接证据", "support": "关联证据", "context": "补充上下文"}


def sentence_lookup(index, result):
    records = index.get("sentences", {}).get("sentences", [])
    by_id = {record.get("id", record.get("sentence_id")): record for record in records}
    for record in result.get("sentence_evidence", {}).get("candidates", []):
        sid = record.get("sentence_id", record.get("id"))
        by_id[sid] = {**by_id.get(sid, {}), **record}
    return by_id


def visible_groups(result, limit=3):
    """Prioritize answer groups and suppress repeated sentence quotations."""
    snapshot, answer = result.get("evidence_groups", {}), result.get("answer", {})
    groups = snapshot.get("groups", [])
    by_id = {group["id"]: group for group in groups}
    answer_ids = list(dict.fromkeys([
        *answer.get("evidence_group_ids", []), *answer.get("group_ids", []),
        *snapshot.get("answer_group_ids", []), *result.get("answer_group_ids", []),
    ]))
    evidence_ids = set(answer.get("evidence_ids", []))
    related_ids = [group["id"] for group in groups if evidence_ids.intersection(group.get("sentence_ids", []))]
    ordered = dict.fromkeys([*answer_ids, *related_ids, *snapshot.get("top_ids", []), *by_id])
    selected, seen = [], set()
    for gid in ordered:
        if gid not in by_id:
            continue
        group = by_id[gid]
        ids = list(dict.fromkeys([*group.get("sentence_ids", []),
                                 *(span["sentence_id"] for span in group.get("source_spans", [])
                                   if span.get("sentence_id"))]))
        unseen = [sid for sid in ids if sid not in seen]
        if not unseen:
            continue
        selected.append({**group, "display_sentence_ids": unseen})
        seen.update(unseen)
        if len(selected) >= limit:
            break
    return selected


def group_sentence_spans(index, result, group):
    """Each entry remains one literal, independently locatable source slice."""
    by_id, records = sentence_lookup(index, result), []
    spans = {span["sentence_id"]: span for span in group.get("source_spans", []) if span.get("sentence_id")}
    display_ids = group.get("display_sentence_ids", group.get("sentence_ids", list(spans)))
    for sid in dict.fromkeys(display_ids):
        record = {**by_id.get(sid, {}), **spans.get(sid, {}), "sentence_id": sid}
        if not isinstance(record.get("source_start"), int) or not isinstance(record.get("source_end"), int):
            continue
        if sid in group.get("context_sentence_ids", []):
            default_role = "context"
        elif sid in group.get("supporting_sentence_ids", []):
            default_role = "support"
        else:
            default_role = "direct"
        record.setdefault("role", default_role)
        record.setdefault("reason", "命中问题内容" if record["role"] == "direct"
                          else "提供对象关联" if record["role"] == "support" else "提供相邻上下文")
        records.append(record)
    return sorted(records, key=lambda record: (record["source_start"], record["source_end"], record["sentence_id"]))


def show_context_groups(index, result, quote_renderer):
    groups = visible_groups(result)
    if not groups:
        return False
    st.markdown("**多句证据组（最多 3 组）**")
    st.caption("逐句保留原文与出处；关联证据用于核验对象关系，补充上下文不自动作为答案依据。重复的原句仅展示一次。")
    answer = result.get("answer", {})
    for rank, group in enumerate(groups, 1):
        st.markdown(f"**{rank:02d} · 证据组 {group['id']}**")
        window_ids = group.get("window_ids", [])
        if window_ids:
            st.caption("召回窗口：" + "、".join(window_ids))
        if isinstance(group.get("coverage"), (int, float)):
            st.caption(f"组内查询内容覆盖度：{group['coverage']:.2%}（不表示答案正确概率）")
        for record in group_sentence_spans(index, result, group):
            sid, loc = record["sentence_id"], record.get("location", {})
            label = ROLE_LABELS.get(record["role"], record["role"])
            st.caption(f"{label} · {sid} · {record.get('chapter') or '未标章节'} · 第 {loc.get('start_line', '?')} 行，"
                       f"第 {loc.get('start_column', '?')} 列起 · 原文字符 {record['source_start'] + 1}–{record['source_end']}")
            quote_html = quote_renderer(index, record, answer.get("fields", {}))
            st.markdown(f'<div class="evidence-text" style="white-space:pre-wrap">{quote_html}</div>', unsafe_allow_html=True)
            st.caption(f"加入原因：{record['reason']}")
            if "<mark>" in quote_html:
                st.caption("高亮标记为答案字段在原文中的位置。")
            with st.expander(f"查看句子原文与定位 · {sid}"):
                st.code(index["source"]["text"][record["source_start"]:record["source_end"]], language=None, wrap_lines=True)
                st.json(record, expanded=False)
        with st.expander(f"查看证据组关联与选择过程 · {group['id']}"):
            details = deepcopy(group)
            details.pop("display_sentence_ids", None)
            st.json(details, expanded=False)
    return True


def show_window_retrieval(index, result):
    if not result or "window_retrieval" not in result:
        return
    windows = result["window_retrieval"]
    candidates, corpus = windows.get("candidates", []), windows.get("corpus", {})
    config = windows.get("config", result.get("config", {}))
    st.subheader("句子窗口检索")
    st.caption("按相邻句子窗口独立计算 BM25，支持跨普通短段落召回。窗口中的邻近句仅提供候选上下文，答案依据见上方证据组。")
    st.caption(f"候选 {windows.get('candidate_count', len(candidates))} 个窗口 · N = {corpus.get('N', 0)} · "
               f"avgdl = {corpus.get('avgdl', 0):.6f} 词项 · k1 = {config.get('k1', '?')}，b = {config.get('b', '?')}")
    if not candidates:
        st.info("当前问题未召回句子窗口。")
    by_id = sentence_lookup(index, result)
    for rank, window in enumerate(candidates[:5], 1):
        wid = window["window_id"]
        st.markdown(f"**{rank:02d} · {wid} · 窗口 BM25 {window['score']:.6f}**")
        st.caption("句子：" + "、".join(window.get("sentence_ids", [])) +
                   " · 原始段落：" + "、".join(window.get("paragraph_ids", [])))
        with st.expander(f"查看窗口原句与计算明细 · {wid}"):
            for sid in dict.fromkeys(window.get("sentence_ids", [])):
                record = by_id.get(sid)
                if record:
                    st.caption(f"{sid} · 原文字符 {record['source_start'] + 1}–{record['source_end']}")
                    st.code(index["source"]["text"][record["source_start"]:record["source_end"]], language=None, wrap_lines=True)
            details = window.get("term_details", [])
            if details:
                st.dataframe([{"检索词项": term["term"], "匹配": "是" if term.get("matched") else "否",
                               "TF": term.get("tf", 0), "窗口 DF": term.get("df", 0),
                               "窗口 IDF": term.get("idf", 0), "权重": term.get("weight", 1),
                               "计分贡献": term.get("contribution", 0)} for term in details],
                             hide_index=True, width="stretch")
            st.json(window, expanded=False)
    with st.expander("窗口查询词、统计与完整召回过程"):
        st.caption("窗口 DF、窗口段长和平均长度单独统计；下方原始段落诊断采用另一套统计，分数不能直接比较。")
        st.code("idf = ln(1 + (N - df + 0.5) / (df + 0.5))\n"
                "L = 1 - b + b × dl / avgdl\n"
                "贡献 = weight × idf × tf × (k1 + 1) / (tf + k1 × L)\n"
                "总分 = 实际计分贡献之和；同组只取最大贡献", language=None)
        st.json(windows, expanded=False)
