"""Reusable source-bound slots and an honest reference answer for every query.

Short answers require a structural match. A lexical hit alone can only become
an explicitly labelled reference response, never a proved factual answer.
"""

from copy import deepcopy
import re

from .general_questions import GENERAL_QUESTION_POLICY, GENERAL_QUESTION_RULES, QUESTION_PATTERNS


GENERAL_POLICY = "source-bound-general-answers/2"
GENERAL_CONFIG = {"max_reference_sentences": 3, "max_slot_chars": 120, "reference_relative_floor": 0.5,
                  "max_subject_chars": 24, "reference_order": "matched sentences and adjacent quoted context; score, source order",
                  "short_answer_requires": "literal structure, complete source, role and scoped state checks",
                  "topic_lookback_sentences": 3}
GENERAL_RULES = {
    "QA101": "身份判断匹配原文的对象是属性结构；评价归属单列；同句前置名词话题可关联紧随的他们等指代，分别保存真实区间。",
    "QA102": "任意动词的显式问句槽位与原文分句按同一已知前后表达匹配；保留角色顺序、限定词、实际词项边界和原文状态。",
    "QA103": "对象的属性是/为/包括/有某值或明确地点表达可以填充字段；字段仅取原文，不把相邻对象的值移给问题对象。",
    "QA104": "贡献、作用、结果等可引用同一句中对象明确且含对应属性或显式结果标记的完整原句；不从结果推导原文未写出的原因。",
    "QA105": "材料、事件地点及方向端点按显式关系填槽；短距离同段落话题继承仅允许相同名词中心语或紧邻的方向结构，并单独保存先行词证据。",
    "QA106": "自然量词保留类别约束；多字段可用逐字段验证且来源连续的完整原句回答，不从相邻词项猜测角色或补写概括。",
    "QAG101": "不完整句、问句、否定、计划、假设、传闻、引语、未解决人物歧义和事实约束不通过通用肯定字段抽取。",
    "QAG102": "同一待答字段存在多个不同原文取值时保留歧义；候选分数和位置不能代替事实绑定证明。",
    "QA201": "所有问法均提供响应；无法验证字段时最多引用三条去重原文，明确标记参考及未解决条件，不把参考材料标为已证明答案。",
    "QA202": "原因或方法问答可将命中句之后、同章节且无标题列表边界的相邻引语作为参考上下文；单独标注来源，不作为字段绑定证明或已召回命中。",
}
GENERAL_PATTERNS = {
    "blocked": r"没有|并未|尚未|未曾|未能|不是|并非|不曾|不(?:会|能|是|送|赠)|未|没|取消|停止|放弃|撤销|计划|打算|准备|原想|拟|将于|将会|将要|即将|可能|或许|也许|预计|预期|有望|如果|假如|假设|倘若|据说|传闻|听说|声称|宣称|[？?]",
    "opinion": r"(?P<speaker>我|我们|[^，,。！？!?；;：:]{1,12}?)(?P<verb>认为|觉得|感到|感觉到|说)(?P<subject>.+)",
    "reference": r"他|她|它|他们|她们|它们|此人|其",
    "result": r"贡献|作用|影响|结果|成就|功劳|更重要|最重要|主要|使得|从而|因此|促进|推动|带来|导致|造成|致使",
    "clause": r"[^，,；;。！？!?：:\n]+",
    "state_description": r"[，,]?(?:仍|还|一直|依然)?(?:保持|保留|呈现|具有|拥有|处于)(?:着|了)?[^，,。！？!?；;]{0,40}",
    "material": r"(?:全部|均|都)?(?:是)?(?:用|由)(?P<value>.+?)(?P<predicate>砌成|制成|建成|做成|制造)(?:的)?",
    "serial_title": r"《[^》]*(?:学报|期刊|杂志|刊|Journal)[^》]*》|(?:期刊|杂志|刊物)[《：:]",
    "governing_prefix": r"^(?:据说|据报道|传闻|听说|如果|假如|假设|倘若)|(?:认为|觉得|表示|说道|说过|曾说|说)[：:]?$",
    "governing_state_header": r"^(?:(?:原定|拟定|未来|后续|今后)?(?:计划|规划|准备|安排)(?:方案|内容|事项|情况)?(?:如下|包括|有|是)?|未来|今后|以后|将来)(?:[，,；;：:]|$)",
    "topic_predicate": r"^(?:是|为|位于|坐落|始建|建于|全长|长|高|重|宽|包含|包括|共|非常|很|十分|格外|特别|具有|拥有|呈现|保持|保留|已经|曾经|正在|在|计划|拟|将)",
}


def general_snapshot():
    return deepcopy({"policy": GENERAL_POLICY, "config": GENERAL_CONFIG, "rules": GENERAL_RULES,
                     "patterns": GENERAL_PATTERNS, "question_policy": GENERAL_QUESTION_POLICY,
                     "question_rules": GENERAL_QUESTION_RULES, "question_patterns": QUESTION_PATTERNS})


def _span(index, sentence, start, end, rule):
    from .extraction import _span as mapped_span
    return mapped_span(index, sentence, start, end, rule)


def _boundary(index, sentence, local):
    absolute = sentence["clean_start"] + local
    return not any(t["clean_start"] < absolute < t["clean_end"]
                   for pid in sentence["paragraph_ids"]
                   for t in index["paragraphs"][int(pid[1:]) - 1]["tokens"])


def _safe(index, result, sentence, fields=None, bindings=()):
    from .answer_safety import blocking_query_constraints, check_fact_scope
    if not sentence["complete"]:
        return "原句不完整或引语边界未通过检查"
    if blocking_query_constraints(result):
        return "问题的事实约束或人物歧义未解决"
    if fields is None:
        return None
    plan = result["question"].get("answer_plan", {})
    scopes, first_clause_start = [], None
    # Validate the owning clause, not an unrelated negative elsewhere in a
    # sentence. A description spans every clause and therefore checks them all.
    for clause in _clauses(sentence["text"]):
        a, b = sentence["clean_start"] + clause.start(), sentence["clean_start"] + clause.end()
        if any(f["clean_start"] < b and a < f["clean_end"] for f in fields.values()):
            scopes.append(clause.group())
            if first_clause_start is None:
                first_clause_start = clause.start()
    prefix = sentence["text"][:first_clause_start or 0]
    if re.search(GENERAL_PATTERNS["governing_prefix"], prefix.rstrip("，,；;：: ")):
        return "前置转述或假设仍限定当前事实"
    if re.search(GENERAL_PATTERNS["governing_state_header"], prefix):
        return "前置计划、准备或未来安排仍限定当前事实"
    # A quote beginning before the field's clause remains a quotation even
    # when its opening punctuation is outside the clause itself.
    if any(prefix.count(left) > prefix.count(right) for left, right in (("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』"))):
        return "当前字段仍位于前置引语范围内"
    # Explicit date/location constraints may live in a sentence-initial adjunct.
    # Include only adjuncts; another subject's clause cannot supply constraints.
    adjunct = re.match(r"^(?:[0-9〇零一二三四五六七八九十年月日时分:：]+)[，,]", sentence["text"])
    if adjunct:
        scopes.insert(0, adjunct.group().rstrip("，,"))
    scope = "，".join(scopes)
    rejection = check_fact_scope(result, scope, allow_report=bool(plan.get("intent") == "identity" and plan.get("attribution")))
    if rejection:
        return rejection
    if re.search(r"谁|什么|何人", scope):
        return "原文仍在提问，不能把问题本身当成答案"
    return None


def _event_actor(index, sentence, plan):
    obj, predicate = plan["object"], plan["predicate"]
    patterns = [r"(?P<value>.+?)" + _literal(predicate) + r"(?:了|过)?" + re.escape(obj),
                re.escape(obj) + r"(?:是由|是被|由|被)(?P<value>.+?)" + _literal(predicate) + r"(?:的)?"]
    for clause in _clauses(sentence["text"]):
        for pattern in patterns:
            match = re.fullmatch(pattern, clause.group())
            if not match:
                continue
            a, b = clause.start() + match.start("value"), clause.start() + match.end("value")
            value = sentence["text"][a:b]
            if (b - a > GENERAL_CONFIG["max_subject_chars"]
                    or re.fullmatch(GENERAL_PATTERNS["reference"], value)
                    or re.search(r"帮助|要求|安排|通知|让|建议|昨天|随后|已经|正在", value)
                    or not (_boundary(index, sentence, a) and _boundary(index, sentence, b))):
                continue
            return {"value": _span(index, sentence, a, b, "QA102")}, [], "fact", "general"
    return None


def _literal(value):
    return re.escape(value.strip())


def _query_literal(result, plan, start, end):
    """Make only *standalone indexed particles* optional, never name substrings."""
    body = plan["body"]
    offset = plan["body_clean_start"]
    particles = {t["clean_start"] - offset: t["clean_end"] - offset
                 for t in result["query_tokens"] if t["term"] in {"了", "着", "过"}
                 and offset + start <= t["clean_start"] and t["clean_end"] <= offset + end}
    pieces, cursor = [], start
    while cursor < end:
        stop = particles.get(cursor, cursor + 1)
        literal = re.escape(body[cursor:stop])
        pieces.append("(?:" + literal + ")?" if cursor in particles else literal)
        cursor = stop
    return "".join(pieces)


def _clauses(text):
    return list(re.finditer(GENERAL_PATTERNS["clause"], text))


def _nominal_topic(index, text):
    if not text or len(text) > GENERAL_CONFIG["max_subject_chars"]:
        return False
    if re.search(r"[0-9]|是|认为|觉得|感到|说|告诉|帮助|要求|批评|赞扬|已经|正在|了|着", text):
        return False
    return (bool(re.fullmatch(r"(?:(?:我们|你们|他们|她们)的|这[些个位名支]|那[些个位名支]|该).{1,16}", text))
            or bool(re.fullmatch(r".{1,10}们", text)) or text in index["name_discovery"]["names"])


def _identity(index, result, sentence, plan):
    clauses = _clauses(sentence["text"])
    predicate = plan.get("predicate") or plan.get("subject")
    if not predicate:
        return None
    # Predicate-first questions can be answered by either literal order.
    for number, clause in enumerate(clauses):
        match = re.fullmatch(r"(?P<subject>.+?)是" + re.escape(predicate), clause.group())
        if not match:
            continue
        subject = match["subject"]
        offset = clause.start() + match.start("subject")
        fields, bindings = {}, []
        opinion = re.fullmatch(GENERAL_PATTERNS["opinion"], subject)
        if opinion:
            # An author's view cannot be supplied from a different speaker.
            if not plan.get("attribution") or opinion["speaker"] not in {"我", "我们"}:
                continue
            end = offset + opinion.start("subject")
            fields["attribution"] = _span(index, sentence, offset, end, "QA101")
            offset += opinion.start("subject")
            subject = opinion["subject"]
        elif re.search(r"认为|觉得|感到|说", subject):
            continue
        value_start, value_end = offset, offset + len(subject)
        if re.fullmatch(GENERAL_PATTERNS["reference"], subject):
            if not number:
                continue
            topic = clauses[number - 1]
            if not _nominal_topic(index, topic.group()):
                continue
            reference = _span(index, sentence, value_start, value_end, "QA101")
            value_start, value_end = topic.span()
            bindings.append({"rule": "QA101", "kind": "same_sentence_topic",
                             "antecedent": _span(index, sentence, value_start, value_end, "QA101"),
                             "reference": reference})
        if not 0 < value_end - value_start <= GENERAL_CONFIG["max_subject_chars"]:
            continue
        if not (_boundary(index, sentence, value_start) and _boundary(index, sentence, value_end)):
            continue
        fields["value"] = _span(index, sentence, value_start, value_end, "QA101")
        return fields, bindings, "opinion" if opinion else "fact", "general"
    # "孔子是谁" can also match "孔子是..." without treating every 是什么
    # question as a formal dictionary definition.
    subject = plan.get("subject")
    if subject:
        return _property(index, sentence, subject, "", "", "QA103")
    return _property(index, sentence, predicate, "", "", "QA103")


def _property(index, sentence, subject, prop, suffix, rule):
    left = re.escape(subject) + (r"(?:的)?" + re.escape(prop) if prop else "")
    connectors = r"(?:是|为|包括|有)" if rule != "QA103-location" else r"(?:位于|坐落于|设在|在)"
    rule = "QA103"
    # An optional comma may separate a nominal subject from its copula, e.g.
    # "最早发现某现象的人，是某研究者". It never licenses another subject.
    pattern = (r"(?:^|[，,；;：:])" + left + r"[，,]?" + connectors
               + r"(?P<value>[^，,；;。！？!?：:]+?)" + _literal(suffix) + r"(?=$|[，,；;。！？!?：:])")
    for match in re.finditer(pattern, sentence["text"]):
        if not match:
            continue
        a, b = match.start("value"), match.end("value")
        if b - a > GENERAL_CONFIG["max_slot_chars"] or not (_boundary(index, sentence, a) and _boundary(index, sentence, b)):
            continue
        return {"value": _span(index, sentence, a, b, rule)}, [], "fact", "general"
    return None


def _topic_binding(index, result, sentence, subject, reference=None):
    """Bounded explicit topic continuity, with real source spans for both ends."""
    from .answer_safety import check_fact_scope
    sentences = index["sentences"]["sentences"]
    position = next((n for n, s in enumerate(sentences) if s["id"] == sentence["sentence_id"]), None)
    if position is None:
        return None
    head = None
    if reference:
        demonstrative = re.fullmatch(r"这(?:么[^的，,]{0,5}的|座|个|条|种|些)?(?P<head>[^的，,]{1,4})", reference.group())
        if not demonstrative or not subject.endswith(demonstrative["head"]):
            return None
        head = demonstrative["head"]
    limit = GENERAL_CONFIG["topic_lookback_sentences"] if head else 1
    for n in range(position - 1, max(-1, position - limit - 1), -1):
        previous = sentences[n]
        if previous["paragraph_ids"] != sentence["paragraph_ids"] or previous.get("chapter") != sentence.get("chapter"):
            break
        if (previous["text"].startswith(subject)
                and re.match(GENERAL_PATTERNS["topic_predicate"], previous["text"][len(subject):])
                and not check_fact_scope(result, previous["text"])):
            previous = {**previous, "sentence_id": previous["id"]}
            if not _boundary(index, previous, len(subject)):
                return None
            return {"kind": "bounded_paragraph_topic", "rule": "QA105",
                    "antecedent": _span(index, previous, 0, len(subject), "QA105"),
                    "reference": _span(index, sentence, *(reference.span() if reference else (0, min(2, len(sentence["text"])))), "QA105")}
        # Every intervening sentence must continue the very same literal head.
        # A new named subject, contrast or paragraph boundary stops inheritance.
        if not head or not previous["text"].startswith(head) or re.search(r"另一|其他|另外|相比|相较", previous["text"]):
            break
    return None


def _material(index, result, sentence, plan):
    subject, predicate = plan["subject"], plan["predicate"]
    clauses = _clauses(sentence["text"])
    for n, clause in enumerate(clauses):
        match = re.fullmatch(r"(?:" + re.escape(subject) + r")?" + GENERAL_PATTERNS["material"], clause.group())
        if not match or match["predicate"] != predicate:
            continue
        bindings = []
        if not clause.group().startswith(subject):
            if n and clauses[n - 1].group() == subject:
                bindings = [{"kind": "same_sentence_topic", "rule": "QA105",
                             "antecedent": _span(index, sentence, *clauses[n - 1].span(), "QA105"),
                             "reference": _span(index, sentence, *clause.span(), "QA105")}]
            elif n and n == 1:
                binding = _topic_binding(index, result, sentence, subject, clauses[0])
                if not binding:
                    continue
                bindings = [binding]
            else:
                continue
        a, b = clause.start() + match.start("value"), clause.start() + match.end("value")
        if not (_boundary(index, sentence, a) and _boundary(index, sentence, b)):
            continue
        return {"value": _span(index, sentence, a, b, "QA105")}, bindings, "fact", "general"
    return None


def _event_location(index, sentence, plan):
    pattern = re.escape(plan["subject"]) + re.escape(plan["predicate"]) + r"(?:在|于|到)(?P<value>.+)"
    for clause in _clauses(sentence["text"]):
        match = re.fullmatch(pattern, clause.group())
        if not match:
            continue
        a, b = clause.start() + match.start("value"), clause.start() + match.end("value")
        if b - a <= GENERAL_CONFIG["max_slot_chars"] and _boundary(index, sentence, a) and _boundary(index, sentence, b):
            return {"value": _span(index, sentence, a, b, "QA105")}, [], "fact", "general"
    return None


def _endpoints(index, result, sentence, plan):
    first, second = plan["first"], plan["second"]
    # Both direction roles must be explicitly present in the source, in order.
    if first == second:
        return None
    subject = plan["subject"]
    prefix = re.escape(subject) + r"[，,]?" if sentence["text"].startswith(subject) else ""
    match = re.fullmatch(prefix + re.escape(first) + r"起(?P<first>[^，,。；;]+)[，,].*?(?:至|到)"
                         + re.escape(second) + r"端(?:的)?(?P<second>[^，,。；;]+)[。]?", sentence["text"])
    if not match:
        return None
    bindings = []
    if not prefix:
        binding = _topic_binding(index, result, sentence, subject)
        if not binding:
            return None
        bindings = [binding]
    fields = {name: _span(index, sentence, *match.span(name), "QA106") for name in ("first", "second")}
    if any(not _boundary(index, sentence, n) for name in ("first", "second") for n in match.span(name)):
        return None
    fields["description"] = _span(index, sentence, 0, len(sentence["text"]), "QA106")
    return fields, bindings, "fact", "excerpt"


def _surface(index, result, sentence, plan):
    slots = plan["slots"]
    if not slots or len(slots) > 3 or plan.get("attribution"):
        return None
    # Existing specialized gift guards still apply to all gift questions.
    if result["question"].get("relation"):
        return None
    body, pieces, cursor = plan["body"], [], 0
    for number, slot in enumerate(slots):
        pieces += [_query_literal(result, plan, cursor, slot["local_start"]),
                   rf"(?P<slot{number}>[^，,。！？!?；;：:]{{1,{GENERAL_CONFIG['max_slot_chars']}}}?)"]
        cursor = slot["local_end"]
    pieces.append(_query_literal(result, plan, cursor, len(body)))
    if not any(body[a:b].strip() for a, b in
               [(0, slots[0]["local_start"]), (slots[-1]["local_end"], len(body))]):
        return None
    pattern = "".join(pieces)
    for clause in _clauses(sentence["text"]):
        match = re.fullmatch(pattern, clause.group())
        if not match:
            continue
        fields = {}
        for number, slot in enumerate(slots):
            a, b = clause.start() + match.start(f"slot{number}"), clause.start() + match.end(f"slot{number}")
            if not (_boundary(index, sentence, a) and _boundary(index, sentence, b)):
                break
            value = sentence["text"][a:b]
            if re.fullmatch(GENERAL_PATTERNS["reference"], value):
                break
            if slot["text"] == "谁" and (len(value) > GENERAL_CONFIG["max_subject_chars"]
                    or re.search(r"昨天|今天|明天|随后|后来|已经|正在|帮助|要求|建议|通知|让|被", value)):
                break
            if re.search(r"期刊|杂志|刊物", slot["text"]) and not re.search(GENERAL_PATTERNS["serial_title"], value, re.I):
                break
            # An explicit plural classifier requires an actual enumeration;
            # the numeral is a constraint on the returned items, not decoration.
            count = re.match(r"哪([两二三四五六七八九十])(?:种|类|本|条|座|处|项)", slot["text"])
            if count:
                expected = "零一二三四五六七八九十".find(count[1].replace("两", "二"))
                if len(re.split(r"、|以及|和|及", value)) != expected:
                    break
            fields[f"slot_{number + 1}"] = _span(index, sentence, a, b, "QA102")
        else:
            if len(slots) == 1:
                fields["value"] = deepcopy(fields["slot_1"])
                return fields, [], "fact", "general"
            fields["description"] = _span(index, sentence, 0, len(sentence["text"]), "QA102")
            return fields, [], "fact", "excerpt"
    return None


def _consequence(index, sentence, plan):
    subject = plan.get("subject")
    if not subject or not sentence["text"].startswith(subject) or not _boundary(index, sentence, len(subject)):
        return None
    if not re.search(GENERAL_PATTERNS["result"], sentence["text"][len(subject):]):
        return None
    return {"description": _span(index, sentence, 0, len(sentence["text"]), "QA104")}, [], "fact", "excerpt"


def _state_description(index, sentence, plan):
    subject, prop = plan["subject"], plan["property"]
    match = re.match(re.escape(subject) + GENERAL_PATTERNS["state_description"] + re.escape(prop), sentence["text"])
    if not match or not _boundary(index, sentence, len(subject)):
        return None
    return {"description": _span(index, sentence, 0, len(sentence["text"]), "QA103")}, [], "fact", "excerpt"


def answer_general(index, result, evidence, base):
    plan = result["question"].get("answer_plan", {})
    if not plan or result["question"]["status"] == "no_content":
        return None
    if result["question"]["type"] in {"time", "quantity", "reason", "method"} and plan["intent"] != "state_description":
        return None
    accepted, decisions = [], []
    for sentence in evidence["candidates"]:
        fields, bindings, statement, template = {}, [], "fact", None
        rejection = _safe(index, result, sentence)
        if not rejection:
            found = None
            if plan["intent"] == "identity":
                found = _identity(index, result, sentence, plan)
            elif plan["intent"] == "event_actor":
                found = _event_actor(index, sentence, plan)
            elif plan["intent"] == "action":
                tail = sentence["text"][len(plan["subject"]):]
                aspect = any(t["term"] in {"了", "过", "正在", "已经", "曾经"}
                             for pid in sentence["paragraph_ids"] for t in index["paragraphs"][int(pid[1:]) - 1]["tokens"]
                             if sentence["clean_start"] + len(plan["subject"]) <= t["clean_start"] < sentence["clean_end"])
                if (sentence["text"].startswith(plan["subject"]) and _boundary(index, sentence, len(plan["subject"]))
                        and aspect and not re.match(r"的|是|为|有|位于|属于", tail)):
                    found = ({"description": _span(index, sentence, 0, len(sentence["text"]), "QA102")}, [], "fact", "excerpt")
            elif plan["intent"] == "attribute":
                found = _property(index, sentence, plan["subject"], plan["property"], "", "QA103")
            elif plan["intent"] == "material":
                found = _material(index, result, sentence, plan)
            elif plan["intent"] == "event_location":
                found = _event_location(index, sentence, plan)
            elif plan["intent"] == "endpoints":
                found = _endpoints(index, result, sentence, plan)
            elif plan["intent"] == "location":
                found = _property(index, sentence, plan["subject"], "", plan["suffix"], "QA103-location")
            elif plan["intent"] == "consequence":
                found = _consequence(index, sentence, plan)
            elif plan["intent"] == "state_description":
                found = _state_description(index, sentence, plan)
            elif plan["intent"] not in {"summary", "description"}:
                found = _surface(index, result, sentence, plan)
            if found:
                fields, bindings, statement, template = found
                rejection = _safe(index, result, sentence, fields, bindings)
                if rejection:
                    fields = {}
            else:
                # Keep a scoped refusal in the diagnostic trail even when the
                # sentence could not fill the structural slot.
                rejection = _safe(index, result, sentence, {"description": _span(index, sentence, 0, len(sentence["text"]), "QAG101")})
        decision = {"accepted": bool(fields), "sentence_id": sentence["sentence_id"],
                    "fields": fields, "reason": "原文结构、待答字段和角色检查通过" if fields else
                    ("QAG101：" + rejection if rejection else "未找到与问题字段绑定的通用结构"),
                    "rule": next(iter(fields.values()))["rule"] if fields else "QAG101" if rejection else "QA102",
                    "bindings": bindings, "statement_kind": statement, "answer_template": template}
        decisions.append(decision)
        if fields:
            accepted.append(decision)
    if not accepted:
        return reference_response(index, result, evidence, {**base, "decisions": base["decisions"] + decisions})
    signatures = {(d["answer_template"], d["statement_kind"],
                   d["fields"]["value" if d["answer_template"] == "general" else "description"]["text"])
                  for d in accepted}
    ids = list(dict.fromkeys(d["sentence_id"] for d in accepted))
    # A negated/uncertain occurrence of the same subject and answer must not
    # disappear just because another candidate supplied a positive assertion.
    by_id = {sentence["sentence_id"]: sentence for sentence in evidence["candidates"]}
    blockers = [d for d in decisions if d["rule"] == "QAG101" and any(
        all(f["matches"] for f in by_id[d["sentence_id"]]["features"])
        and a["fields"].get("value", a["fields"].get("description", {})).get("text", "")
            in by_id[d["sentence_id"]]["text"] for a in accepted)]
    ids = list(dict.fromkeys([*ids, *(d["sentence_id"] for d in blockers)]))
    if len(signatures) != 1 or blockers:
        return {**base, "status": "ambiguous", "mode": "conflicting_fields", "fields": {},
                "evidence_ids": ids, "decisions": base["decisions"] + decisions,
                "text": "原文中找到多个不同的候选回答，请分别核对下方证据。",
                "reason": "QAG102：同一待答字段存在多个取值，当前规则不擅自选择或合并。", "rule_ids": ["QAG102"]}
    selected = accepted[0]
    kind = "opinion" if selected["statement_kind"] == "opinion" else selected["answer_template"]
    template = index["answering"]["templates"][kind]
    return {**base, "status": "answered", "mode": selected["answer_template"],
            "statement_kind": selected["statement_kind"], "fields": selected["fields"],
            "bindings": selected["bindings"], "template_id": template["id"], "template": template,
            "text": template["text"].format(**{key: value["text"] for key, value in selected["fields"].items()}),
            "evidence_ids": list(dict.fromkeys([selected["sentence_id"], *(b["antecedent"]["sentence_id"] for b in selected["bindings"] if b.get("antecedent"))])),
            "decisions": base["decisions"] + decisions,
            "reason": "通用结构与待答字段匹配，每个答案字段均可定位至原文。",
            "rule_ids": [selected["rule"], "G104"]}


def reference_response(index, result, evidence, answer):
    """Provide actual source content without inflating the proved-answer rate."""
    if answer["status"] in {"answered", "ambiguous", "no_content"}:
        return answer
    references, seen = [], set()
    matched = [s for s in evidence["candidates"] if s.get("matched")]
    if matched:
        floor = matched[0]["score"] * GENERAL_CONFIG["reference_relative_floor"]
        matched = [s for s in matched if s["score"] >= floor]
        statements = [s for s in matched if not re.search(r"[？?]|谁|什么", s["text"])]
        matched = statements or matched
    ordered = [(s, "matched_sentence") for s in matched]
    # Quote boundaries remain strict for facts. Only reference responses may
    # display a nearby quotation, with a separate and replayable provenance.
    if result["question"].get("answer_plan", {}).get("intent") in {"reason", "method"}:
        sentences = index["sentences"]["sentences"]
        positions = {s["id"]: n for n, s in enumerate(sentences)}
        boundaries = {(b["left_sentence_id"], b["right_sentence_id"]): set(b["reasons"])
                      for b in index["context"]["boundaries"]}
        quoted_pairs = []
        for sentence in matched:
            n = positions[sentence["sentence_id"]] + 1
            if n >= len(sentences):
                continue
            neighbor = sentences[n]
            blocked = boundaries.get((sentence["sentence_id"], neighbor["id"]), set()) - {"CW005"}
            gap = index["cleaning"]["text"][sentence["clean_end"]:neighbor["clean_start"]]
            if (blocked or gap.strip() or sentence.get("chapter") != neighbor.get("chapter")
                    or not re.search(r'''[“”「」"']''', neighbor["text"])):
                continue
            quoted_pairs += [(sentence, "matched_sentence"),
                             ({**neighbor, "sentence_id": neighbor["id"]}, "adjacent_reference_only")]
        ordered = quoted_pairs + ordered
    for sentence, origin in ordered:
        text = index["source"]["text"][sentence["source_start"]:sentence["source_end"]]
        if text in seen:
            continue
        seen.add(text)
        references.append({"sentence_id": sentence["sentence_id"], "text": text,
                           "source_start": sentence["source_start"], "source_end": sentence["source_end"],
                           "complete": sentence["complete"], "origin": origin,
                           "location": deepcopy(sentence["location"])})
        if len(references) == GENERAL_CONFIG["max_reference_sentences"]:
            break
    if not references:
        return {**answer, "status": "insufficient_evidence", "mode": "no_evidence",
                "text": "当前文本的检索范围内，没有找到足以回答这个问题的原文。可以补充文档或明确问题中的对象与事件。",
                "reason": "检索没有命中可引用的内容；不能据此断言全文不存在答案。"}
    intent = result["question"].get("answer_plan", {}).get("intent")
    lead = "针对这个问题，检索到以下原文，可作为回答参考："
    if intent == "summary":
        lead = "下面摘录与问题相关的原文要点，供概括或分析时参考："
    reason = answer.get("reason", "")
    rejected = list(dict.fromkeys(d.get("reason", "") for d in answer.get("decisions", [])
                                 if not d.get("accepted") and d.get("reason")))
    if rejected:
        reason = "；".join(rejected[:3])
    text = lead + "\n\n" + "\n\n".join(f"{n}. {row['text']}" for n, row in enumerate(references, 1))
    text += "\n\n以上为原文参考，尚未确认它们完整回答了问题。"
    return {**answer, "status": "evidence_only", "mode": "reference", "text": text,
            "reason": reason or "尚未建立问题与答案字段之间的可靠绑定。", "references": references,
            "evidence_ids": [row["sentence_id"] for row in references if row["origin"] == "matched_sentence"],
            "reference_context_ids": [row["sentence_id"] for row in references if row["origin"] == "adjacent_reference_only"],
            "rule_ids": list(dict.fromkeys([*answer.get("rule_ids", []), "QA201",
                         *(["QA202"] if any(r["origin"] == "adjacent_reference_only" for r in references) else [])]))}
