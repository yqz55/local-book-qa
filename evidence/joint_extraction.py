"""Narrow, source-bound answers from two adjacent sentences.

Only a named project/trial/activity/experiment/engineering work followed by
``该<same type>`` is supported.  No rewritten sentence or invented source span
is used: antecedent, reference and answer each retain their real source range.
"""

import re


JOINT_POLICY = "adjacent-explicit-anaphora/1"
JOINT_RULES = {
    "J101": "连续两句中，前句以问题中明确命名的唯一项目/试点/活动/实验/工程为主语，后句以同类型的该X开头；唯一日期与所问事件在后句的显式句式中绑定。",
    "J102": "同一受限两句指代规则中，后句的明确属性与问题属性一致，抽取唯一原文数量并核对指定单位。",
    "JG101": "两句须完整且为肯定事实；否定、计划、假设、传闻、引语及复杂施受事使联合抽取停止。",
    "JG102": "仅接受文档顺序直接相邻且都位于本次检索范围的两句；章节、标题、列表边界、不唯一前件或截取更长词项内部的名称/指代使联合抽取停止。",
    "JG103": "多个日期/数值或不能由明确句式唯一绑定的字段不得选取第一个；字段和绑定证明分别保存真实原文区间。",
}

_TYPES = r"项目|试点|活动|实验|工程"
_SUBJECT = re.compile(r"(?P<name>[A-Za-z0-9\u3400-\u9fff]{1,48}?(?P<type>" + _TYPES + r"))(?P<tail>.*)")
_REFERENCE = re.compile(r"(?P<reference>该(?P<type>" + _TYPES + r"))")
_EVENTS = ("启动", "开始", "结束", "发布", "举办", "召开", "成立", "建成", "竣工", "投用", "完成", "开工", "上线", "实施", "开展", "发射")
_PROPERTIES = ("覆盖面积", "参与人数", "参加人数", "总投资额", "总投资", "投资额", "总长度", "持续时间", "预算", "容量", "面积", "投资", "数量", "人数", "长度", "重量", "费用", "成本", "经费", "金额")
_UNITS = ("平方千米", "平方公里", "平方米", "立方米", "千米/秒", "千米", "公里", "厘米", "毫米", "公斤", "千克", "毫克", "毫升", "小时", "分钟", "个人", "余人", "万人", "万元", "亿元", "摄氏度", "克", "米", "人", "个", "本", "份", "件", "台", "次", "天", "秒", "元", "升", "%", "％", "℃")
_INTRO = re.compile(r"(?:(?:目前|现已|已经|正式|成功))?(?:进入|启动(?!仪式|日期|时间|地点|工作|情况|负责人)|开始(?!日期|时间)|完成(?!日期|时间)|建成|成立|位于|设在|覆盖|包括|涉及|采用|是|为|实施|开展|开工|上线|投用|举办(?!日期|时间|地点)|召开(?!日期|时间|地点))")
_MEDIATION = re.compile(r"帮助|介绍|要求|建议|安排|通知|祝贺|负责|协助|让|被|把|向|对于|关于|新闻|报道|消息|讨论|提到|强调")
_COORDINATION = re.compile(r"和|与|及|、|以及|同时|分别|另一|其他|其它")


def _id(sentence):
    return sentence.get("sentence_id") or sentence["id"]


def _query_plan(result):
    """A literal named subject plus one supported requested predicate."""
    question = result["question"]
    matched = _SUBJECT.fullmatch(question.get("target", ""))
    if not matched or len(re.findall(_TYPES, matched["name"])) != 1:
        return None
    name, object_type, tail = matched["name"], matched["type"], matched["tail"]
    if _COORDINATION.search(name) or name.startswith(("该", "这", "那", "另一")):
        return None
    if question["type"] == "time":
        for event in _EVENTS:
            if re.fullmatch(r"(?:正式|成功)?" + event + r"(?:时间|日期)?", tail):
                return name, object_type, event, None
    if question["type"] == "quantity":
        for prop in _PROPERTIES:
            if tail == prop:
                return name, object_type, prop, None
            for unit in _UNITS:
                if tail == prop + unit:
                    return name, object_type, prop, unit
    return None


def _blocked(index, left, right):
    reasons = []
    if left.get("chapter") != right.get("chapter"):
        reasons.append("chapter")
    for boundary in index.get("context", {}).get("boundaries", []):
        if (boundary.get("left_sentence_id") == _id(left)
                and boundary.get("right_sentence_id") == _id(right)):
            reasons.extend(boundary.get("reasons") or ["context_boundary"])
    # Also safe on older indexes without a context snapshot.  A heading/list
    # omitted by the sentence builder leaves non-whitespace in the gap.
    gap = index["cleaning"]["text"][left["clean_end"]:right["clean_start"]]
    if gap.strip():
        reasons.append("intervening_text")
    for sentence in (left, right):
        if re.match(r"\s*(?:[-*•●]|[0-9]+[.、）)]|[（(][0-9]+[）)])\s*", sentence["text"]):
            reasons.append("list_item")
    return sorted(set(reasons))


def _field(index, sentence, start, end, rule):
    # Import lazily: extraction.py integrates this module after its own helpers
    # are defined.  _span maps each real sentence range independently.
    from .extraction import _span
    return _span(index, {**sentence, "sentence_id": _id(sentence)}, start, end, rule)


def _token_boundary(index, sentence, local_position):
    """A literal object/reference cannot stop inside a longer indexed word."""
    position = sentence["clean_start"] + local_position
    paragraph_ids = set(sentence["paragraph_ids"])
    return not any(token["clean_start"] < position < token["clean_end"]
                   for paragraph in index["paragraphs"] if paragraph["id"] in paragraph_ids
                   for token in paragraph["tokens"])


def extract_joint(index, result, evidence):
    """Return accepted and rejected relevant adjacent-pair decisions.

    ``evidence['candidates']`` defines the eligible sentence scope.  A matching
    antecedent/reference pair is auditable even when rejected.  In particular
    JG101 on a matching predicate is surfaced for the final conflict check.
    """
    plan = _query_plan(result)
    if plan is None or result["question"]["type"] not in {"time", "quantity"}:
        return []
    name, object_type, predicate, query_unit = plan
    kind = result["question"]["type"]
    query_tail = result["question"].get("target", "")[len(name):]
    requested_modifier = next((modifier for modifier in ("正式", "成功")
                               if kind == "time" and query_tail.startswith(modifier)), None)
    rule = "J101" if kind == "time" else "J102"
    patterns = index["answering"]["patterns"]
    eligible = {_id(s) for s in evidence["candidates"]}
    sentences = index["sentences"]["sentences"]
    decisions = []
    for left, right in zip(sentences, sentences[1:]):
        if _id(left) not in eligible or _id(right) not in eligible:
            continue
        # The named subject must actually be at the beginning, and the reference
        # must be first in the next sentence, not inside another clause.
        if not left["text"].startswith(name):
            continue
        reference = _REFERENCE.match(right["text"])
        if not reference:
            continue
        subject = _field(index, left, 0, len(name), rule)
        ref = _field(index, right, *reference.span("reference"), rule)
        boundary_reasons = _blocked(index, left, right)
        remainder = left["text"][len(name):]
        unique = (len(re.findall(_TYPES, left["text"])) == 1
                  and len(re.findall(_TYPES, right["text"])) == 1
                  and not _COORDINATION.search(left["text"]))
        leading_date = re.match(r"(?:于|在)" + patterns["date"], remainder)
        explicit_subject = bool(_INTRO.match(remainder) or
                                (leading_date and _INTRO.match(remainder[leading_date.end():])))
        lexical_boundaries = (_token_boundary(index, left, len(name))
                              and _token_boundary(index, right, reference.end()))
        same_type = reference["type"] == object_type
        body = right["text"][reference.end():].rstrip("。.!！")
        relevant = (bool(re.search(re.escape(predicate) + r"(?:了)?[。.!！?？]?$", right["text"])
                         or re.search(re.escape(predicate) + r"(?:于|在)" + patterns["date"] + r"[。.!！?？]?$", right["text"])) if kind == "time"
                    else body.startswith((predicate, "的" + predicate)))
        checks = {"adjacent_in_document": True, "both_in_retrieval_scope": True,
                  "boundary_reasons": boundary_reasons, "single_explicit_antecedent": unique,
                  "explicit_subject_clause": explicit_subject,
                  "actual_term_boundaries": lexical_boundaries,
                  "matching_reference_type": same_type, "matching_requested_predicate": relevant,
                  "complete_sentences": bool(left.get("complete") and right.get("complete"))}
        if requested_modifier:
            checks["required_modifier"] = requested_modifier
            checks["matching_event_modifier"] = (not requested_modifier
                                                  or requested_modifier + predicate in body)
        binding = {"rule": rule, "antecedent": subject, "reference": ref, "checks": checks}
        decision = {"accepted": False, "rule": rule, "reason": "", "sentence_ids": [_id(left), _id(right)],
                    "fields": {}, "binding": binding}
        decisions.append(decision)

        def reject(code, reason):
            decision["rule"] = code
            decision["reason"] = code + "：" + reason

        if result["question"].get("constraints") or result.get("query_ambiguities"):
            reject("JG102", "问题含尚未支持的事实约束或人物歧义")
            continue
        if boundary_reasons or not same_type or not unique or not lexical_boundaries:
            reject("JG102", "边界、同类型指代或唯一明确前件检查未通过")
            continue
        if not relevant:
            reject("JG102", "后句未明确涉及所问事件或属性")
            continue
        if kind == "time" and not checks.get("matching_event_modifier", True):
            reject("JG102", "后句未保留问题要求的事件修饰词")
            continue
        unsafe = []
        for sentence in (left, right):
            if not sentence.get("complete"):
                unsafe.append("incomplete_sentence")
            if re.search(patterns["unsafe"], sentence["text"]):
                unsafe.append("nonfactual_or_negative")
            if re.search(patterns["quote"], sentence["text"]):
                unsafe.append("quotation")
            if _MEDIATION.search(sentence["text"]):
                unsafe.append("mediating_event_or_role")
        checks["unsafe_reasons"] = sorted(set(unsafe))
        if unsafe:
            reject("JG101", "含不完整句、否定、非事实状态、引语或复杂施受事，保留原文核对")
            continue
        if not explicit_subject:
            reject("JG102", "前句未命中命名对象作为显式主语的受支持句式")
            continue
        pattern = patterns["date" if kind == "time" else "quantity"]
        numeric = list(re.finditer(pattern, body))
        if len(numeric) > 1:
            reject("JG103", "后句有多个日期或数值，无法唯一绑定")
            continue
        if kind == "time":
            date = r"(?P<answer>" + pattern + r")"
            pred = re.escape(predicate)
            form = (re.fullmatch(r"(?:已|已经)?(?:于|在)?" + date + r"(?:正式|成功)?" + pred + r"(?:了)?", body)
                    or re.fullmatch(r"(?:已|已经)?(?:正式|成功)?" + pred + r"(?:于|在)" + date, body))
        else:
            form = re.fullmatch(r"(?:的)?" + re.escape(predicate) + r"(?:为|是|达到|达|共计|合计|有)?(?P<answer>" + pattern + r")", body)
        if not form or len(numeric) != 1:
            reject("JG103" if numeric else "JG102", "未命中唯一字段与所问事件或属性直接绑定的受支持句式")
            continue
        if kind == "quantity":
            from .extraction import _requested_unit
            requested = query_unit or _requested_unit(result)
            checks["requested_unit"] = requested
            if requested and not re.search(r"[0-9零〇一二三四五六七八九十百千万亿两]\s*" + re.escape(requested) + r"(?:以上|以下|左右|以内|以外)?$", form["answer"]):
                reject("JG102", "原文数量单位与问题指定单位不一致")
                continue
            if any(m.start() < form.end("answer") and form.start("answer") < m.end()
                   for m in re.finditer(patterns["date"], body)):
                reject("JG102", "日期中的数字不能作为所问数量")
                continue
        value = _field(index, right, reference.end() + form.start("answer"),
                       reference.end() + form.end("answer"), rule)
        decision.update(accepted=True, fields={"subject": subject, "reference": ref,
                                               "time" if kind == "time" else "value": value},
                        reason="相邻两句的唯一明确前件、同类型指代和所问字段绑定检查通过")
    return decisions
