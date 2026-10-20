"""Validate the *bound fact*, rather than unrelated text in its sentence.

Callers remain responsible for proving the subject, predicate and field spans.
They pass only those clauses which belong to that fact; this module never
rescues an otherwise unbound value by finding a date elsewhere in a document.
"""

import re


SAFETY_POLICY = "bound-fact-state/1"


_PREDICATE = (r"通过|采用|利用|使用|于|在|将|把|送|赠|启动|开放|发射|开展|完成|实施|"
              r"建设|建造|成立|建立|发布|举办|召开|采集|发现|研究|投入|运行|生产|"
              r"增加|减少|达到|调整|安装|采购|搬迁|取消|停止|公布|推出|返回|出发|"
              r"修缮|维修|修复|改造|测试|验证|检验|调查|测量|更新|升级|招募|招聘|[0-9]")
# 计划 is also a noun. Require a following action/temporal expression rather
# than rejecting factual phrases such as 计划的内容、计划是、计划经济.
_PLAN_GRAMMAR = r"计划(?=[\u3400-\u9fff0-9])(?!经济|名称|内容|方案|要求|目标|指标|人数|数量|规模|预算|费用|书|表|性|[是为的有])"
_PREPARE_GRAMMAR = r"准备(?=[\u3400-\u9fff0-9])(?!不足|充分|情况|阶段|程度|状态|[是为的])"
_STATES = {
    "planned": rf"原本打算|原想|打算|(?:想|要|欲|愿|试图|尝试)(?=把|将|送|赠)|(?:计划|规划|安排)(?:如下|是|为)|{_PLAN_GRAMMAR}|{_PREPARE_GRAMMAR}|拟(?=[\u3400-\u9fff0-9])",
    "future": r"将要|即将|将会|将于|将(?=[\u3400-\u9fff0-9])",
    "hypothetical": r"如果|假如|倘若|假设(?!检验|验证|分析)",
    "uncertain": rf"可能|或许|也许|预计(?={_PREDICATE}|明年|未来)|预期(?={_PREDICATE}|明年|未来)|预定(?={_PREDICATE}|明年|未来)|有望|力争|是否|能否",
    "completed": r"已经|现已|已(?=经|于|在|完成|启动|成立|发现|发布|采集|投入|达到)",
    "past": r"曾经|曾(?=于|在|把|将|经|完成|启动|成立|发现|发布|采集|是|有)",
    "ongoing": r"正在",
    "prescriptive": r"应该|应当|必须|须(?=[\u3400-\u9fff])",
}
_QUOTES = re.compile(r'''[“”‘’「」『』"']''')
_REPORT = re.compile(r"据说|传闻|听说|声称|宣称|认为|觉得|表示|报道|说道|说过|曾说|说[：:]")
_NEGATIVE = re.compile(r"没有|并未|尚未|未曾|未能|不曾|不是|并非|不会|不能|无法|取消|停止|放弃|撤销|"
                       r"未(?!来|央)|没|不(?!超过|低于|小于|大于|足|少于|到(?=[0-9零〇一二三四五六七八九十百千万])|多于(?=[0-9零〇一二三四五六七八九十百千万])|仅|但|同|锈|丹)")


def safety_snapshot():
    return {"policy": SAFETY_POLICY, "states": dict(_STATES),
            "predicate_grammar": _PREDICATE, "plan_grammar": _PLAN_GRAMMAR,
            "prepare_grammar": _PREPARE_GRAMMAR, "quote": _QUOTES.pattern,
            "report": _REPORT.pattern, "negative": _NEGATIVE.pattern,
            "scope": "Only structurally bound facts and their governing context; preserve source polarity and modality."}


def status_grammar(text, start, end, value):
    """A short state marker needs predicate grammar, not a character in a name."""
    marker = text[start:end]
    if value == "planned" and marker == "拟":
        # This character also starts compound nouns. A query cannot discard
        # an unknown nominal prefix merely because it resembles a modal.
        return bool(re.match(rf"拟(?={_PREDICATE})", text[start:]))
    if value == "planned" and marker == "计划":
        return bool(re.match(_PLAN_GRAMMAR, text[start:]))
    if value == "planned" and marker == "准备":
        return bool(re.match(_PREPARE_GRAMMAR, text[start:]))
    return True


def blocking_query_constraints(result):
    """Return constraints that cannot be established by a local factual clause."""
    issues = []
    known_states = {"modal", "past", "completed", "ongoing", "planned", "future"}
    for row in result.get("question", {}).get("constraints", []):
        kind, value = row.get("kind"), row.get("value")
        if kind in {"time", "location", "negation"}:
            continue
        if kind == "status" and value in known_states:
            continue
        issues.append("unsupported_query_" + str(kind) + "_" + str(value))
    if result.get("query_ambiguities"):
        issues.append("query_name_ambiguity")
    return sorted(set(issues))


def _literal_constraint(value, text):
    # Prevent 2020年 from matching 12020年, and 5月 from matching 15月.
    value = re.sub(r"\s+", "", value)
    text = re.sub(r"\s+", "", text)
    edge = r"(?<![0-9〇零一二三四五六七八九十百千万])" if value[:1].isdigit() else ""
    return bool(re.search(edge + re.escape(value), text))


def _source_status(result, text, state):
    for match in re.finditer(_STATES[state], text):
        marker = match.group()
        if marker == "将":
            # 将 also introduces an object in disposal constructions. Keep
            # their visible object/result grammar (including gifts).
            tail = text[match.end():]
            if (re.match(r"[^，,；;。！？!?]{1,30}(?:送给|赠予|赠与|赠给|转化为|转为|视为|定义为|作为|储存|分成)", tail)
                    or text[:match.start()].endswith(("是", "指", "包括", "用于"))):
                continue
        if marker == "拟":
            # A literal nominal modifier before 的 supplies a reusable name
            # boundary without maintaining a list of species or product names.
            question = result.get("question", {})
            if not any(c.get("value") == "planned" for c in question.get("constraints", [])):
                raw_query = "".join(d["text"] for d in question.get("decisions", [])
                                    if d.get("action") == "retained")
                body = question.get("answer_plan", {}).get("body", raw_query)
                nominal = re.search(r"拟[^，,；;。！？!?的]{2,24}(?=的)", body)
                if nominal and text[match.start():].startswith(nominal.group()):
                    continue
        return True
    return False


def check_fact_scope(result, text, *, allow_report=False):
    """Return a refusal reason, or ``None`` for a locally verified fact scope.

    Normal past-tense wording does not require the source to repeat 曾. Strong
    aspect claims (已经/正在) and explicit event plans must be visibly preserved.
    A future/plan question can only use a source with that same status, never an
    unconditional assertion from which a plan is inferred.
    """
    blocked = blocking_query_constraints(result)
    if blocked:
        return blocked[0]
    if not text or _QUOTES.search(text):
        return "quotation_or_empty_fact_scope"
    if re.search(r"[？?]|谁|什么|何人", text):
        return "source_question_not_fact"
    if not allow_report and _REPORT.search(text):
        return "reported_fact_not_verified"
    constraints = result.get("question", {}).get("constraints", [])
    query_states = {c.get("value") for c in constraints if c.get("kind") == "status"}
    for c in constraints:
        kind, value = c.get("kind"), c.get("value")
        if kind in {"time", "location"} and not _literal_constraint(str(value), text):
            return "unverified_query_" + kind
        if kind == "status" and value in {"completed", "ongoing", "planned", "future"}:
            if not _source_status(result, text, value):
                return "unverified_query_" + value
        if kind == "negation" and not _NEGATIVE.search(text):
            return "unverified_query_negation"
    for state in ("hypothetical", "uncertain", "planned", "future", "prescriptive"):
        if _source_status(result, text, state) and state not in query_states:
            return "source_" + state + "_not_fact"
    if _NEGATIVE.search(text):
        return "source_negative_or_cancelled"
    return None
