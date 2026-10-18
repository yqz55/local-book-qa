"""Display deterministic answers and their exact sentence evidence."""

import html

import streamlit as st

from .context_evidence import show_context_groups


ANSWER_STATES = {
    "answered": "已形成规则答案",
    "evidence_only": "找到相关证据，暂未形成规则答案",
    "insufficient_evidence": "当前检索范围内证据不足",
    "unsupported": "当前问法暂不支持规则作答",
    "ambiguous": "候选证据存在歧义或冲突",
    "no_content": "问题缺少可检索的内容",
}
QUESTION_TYPES = {
    "definition": "定义", "time": "时间", "quantity": "数量",
    "relation": "人物或事件关系", "description": "原文描述",
    "reason": "原因", "method": "方法", "keywords": "关键词检索",
    "unsupported": "暂不支持的问法",
    "identity": "身份或评价对象", "attribute": "属性", "location": "地点",
    "consequence": "作用或结果", "general": "通用字段问答", "summary": "概括或分析",
    "event_actor": "事件参与者", "action": "行为或活动",
    "state_description": "属性或状态描述",
}
CONSTRAINT_TYPES = {"negation": "否定", "status": "事实状态", "time": "时间", "location": "地点"}
DISCOURSE_FIELDS = {"reason": "原文原因", "effect": "所述结果", "method": "原文方法", "target": "对应对象",
                    "subject": "原文主体", "purpose": "原文目的", "modality": "原文语气",
                    "prescription": "原文操作说明", "reference": "原文承接表达"}


def _span_fields(value):
    """Yield only explicitly mapped fields, including nested rule output."""
    if isinstance(value, dict):
        if isinstance(value.get("source_start"), int) and isinstance(value.get("source_end"), int):
            yield value
        else:
            for item in value.values():
                yield from _span_fields(item)
    elif isinstance(value, list):
        for item in value:
            yield from _span_fields(item)


def sentence_quote(index, candidate, fields):
    """Return escaped raw-source HTML with answer spans highlighted.

    Candidate text is cleaned text. Evidence quotations must use source offsets
    instead, so whitespace and punctuation remain exactly as imported.
    """
    raw = index["source"]["text"]
    start, end = candidate["source_start"], candidate["source_end"]
    quote = raw[start:end]
    ranges = []
    for field in _span_fields(fields):
        left, right = field["source_start"], field["source_end"]
        if start <= left < right <= end:
            ranges.append((left - start, right - start))
    merged = []
    for left, right in sorted(ranges):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    parts, cursor = [], 0
    for left, right in merged:
        parts.extend((html.escape(quote[cursor:left]), "<mark>", html.escape(quote[left:right]), "</mark>"))
        cursor = right
    parts.append(html.escape(quote[cursor:]))
    return "".join(parts)


def _keyword_text(keywords):
    return "、".join(str(item.get("term", item.get("text", item))) if isinstance(item, dict) else str(item)
                    for item in keywords)


def discourse_source_fields(index, answer):
    """Expose literal cause/method fields with their independent source ranges."""
    raw, fields = index["source"]["text"], answer.get("fields", {})
    if answer.get("status") != "answered" or not (fields.get("reason") or fields.get("method")):
        return []
    rows = []
    for name, label in DISCOURSE_FIELDS.items():
        for field in _span_fields(fields.get(name)):
            start, end = field["source_start"], field["source_end"]
            if not 0 <= start < end <= len(raw):
                continue
            rows.append({"字段": label, "原文内容": raw[start:end], "来源句": field.get("sentence_id", "未标句号"),
                         "原文字符": f"{start + 1}–{end}"})
    return rows


def show_discourse_fields(index, answer):
    rows = discourse_source_fields(index, answer)
    if not rows:
        return
    st.markdown("**答案字段与原文出处**")
    if answer.get("statement_kind") == "procedure":
        st.caption("原文给出操作方法或步骤；步骤说明不代表这些操作已经执行。下方字段逐项保留原文，证据中的高亮与字段对应。")
    elif answer.get("fields", {}).get("reason"):
        st.caption("原因关系取自原文明确表述；仅引用已标明的原因与结果，不补充推测。证据中的高亮与下方字段对应。")
    else:
        st.caption("方法取自原文明确表述，并保留原文的事实状态。证据中的高亮与下方字段对应。")
    for row in rows:
        st.caption(f"{row['字段']} · {row['来源句']} · 原文字符 {row['原文字符']}")
        st.text(row["原文内容"])


def show_answer(index, result):
    if not result or not result.get("answer"):
        return
    answer = result["answer"]
    question = result.get("question", {})
    sentences = result.get("sentence_evidence", {})
    st.subheader("回答与原文证据" if result.get("general_answer_policy") else "规则答案与句子证据")
    state = answer.get("status", "evidence_only")
    label = ANSWER_STATES.get(state, state)
    if answer.get("mode") == "reference":
        label = "原文参考回答"
    elif answer.get("mode") == "excerpt":
        label = "已形成原句答案"
    if state == "answered":
        st.success(label)
    elif state == "ambiguous":
        st.warning(label)
    else:
        st.info(label)
    if answer.get("text"):
        # Treat imported source text as plain text, including template fields.
        st.text(answer["text"])
    if answer.get("reason"):
        st.caption(answer["reason"])
    if answer.get("mode") == "reference":
        st.caption("已提供实际检索到的原文；这些材料尚未通过完整答案的字段绑定检查。")
        with st.expander("原文参考的逐条出处"):
            for row in answer.get("references", []):
                origin = "命中句后的相邻引语，仅供参考" if row.get("origin") == "adjacent_reference_only" else "检索命中原句"
                st.caption(f"{row['sentence_id']} · {origin} · 原文字符 {row['source_start'] + 1}–{row['source_end']}")
                st.text(row["text"])
    if answer.get("statement_kind") == "opinion":
        st.caption("此回答保留原文的评价归属，表示作者在文中的观点。")
    show_discourse_fields(index, answer)

    kind = question.get("answer_plan", {}).get("intent", question.get("type", "keywords"))
    plan = question.get("answer_plan", {})
    target = plan.get("subject") or plan.get("predicate") or question.get("target") or "未识别"
    st.caption(f"提问意图：{QUESTION_TYPES.get(kind, kind)} · 目标：{target}")
    st.text(f"实际检索关键词：{_keyword_text(question.get('keywords', [])) or '无'}")
    filtered = list(dict.fromkeys(item.get("text", "") for item in question.get("decisions", [])
                                  if item.get("action") == "filtered" and item.get("text")))
    if filtered:
        st.text(f"不参与内容检索的表达：{'、'.join(filtered)}")
    constraints = question.get("constraints", [])
    if constraints:
        st.text("保留的事实约束：" + "；".join(
            f"{CONSTRAINT_TYPES.get(item.get('kind'), '约束')}：{item.get('text', item.get('value', ''))}"
            if isinstance(item, dict) else str(item) for item in constraints))
    else:
        st.caption("未识别到额外的事实约束。")

    grouped = show_context_groups(index, result, sentence_quote)
    candidates = sentences.get("candidates", [])
    by_id = {candidate["sentence_id"]: candidate for candidate in candidates}
    top_ids = sentences.get("top_ids", [candidate["sentence_id"] for candidate in candidates])
    visible_ids = list(dict.fromkeys([*answer.get("evidence_ids", []), *top_ids]))
    visible = [by_id[sid] for sid in visible_ids if sid in by_id][:3]
    if visible and not grouped:
        st.markdown("**句子证据（最多 3 条）**")
    for rank, candidate in enumerate([] if grouped else visible, 1):
        sid = candidate["sentence_id"]
        loc = candidate.get("location", {})
        chapter = candidate.get("chapter") or "未标章节"
        st.caption(f"{rank:02d} · {sid} · {chapter} · 第 {loc.get('start_line', '?')} 行，第 "
                   f"{loc.get('start_column', '?')} 列起 · 原文字符 {candidate['source_start'] + 1}–{candidate['source_end']}")
        quote_html = sentence_quote(index, candidate, answer.get("fields", {}))
        st.markdown(f'<div class="evidence-text" style="white-space:pre-wrap">{quote_html}</div>', unsafe_allow_html=True)
        st.caption("高亮标记为答案字段在原文中的位置。" if "<mark>" in quote_html else "引文按原文字符区间展示。")
        with st.expander(f"查看句子原文与定位 · {sid}"):
            st.code(index["source"]["text"][candidate["source_start"]:candidate["source_end"]],
                    language=None, wrap_lines=True)
            st.json(candidate, expanded=False)

    with st.expander("问题提炼、句子排序与答案规则明细"):
        st.caption("记录实际关键词选择、事实约束、候选句排序和抽取过程；排序分数不表示答案正确概率。")
        st.json({"question": question, "sentence_evidence": sentences,
                 "evidence_groups": result.get("evidence_groups"), "answer": answer}, expanded=False)

