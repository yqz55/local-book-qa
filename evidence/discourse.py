"""Explicit, source-bound causal and method relations.

These rules identify a relation already written in the source.  They do not
infer causes from purpose, time order, correlation or general world knowledge.
Every returned field is one genuine interval in one source sentence.
"""

import re


DISCOURSE_POLICY = "explicit-cause-method/1"
DISCOURSE_CONFIG = {"max_sentences": 2, "max_method_chars": 240,
                    "anchor_order": "literal query token order; actual indexed token spans",
                    "procedure_answer": "source prescription; no assertion of completion"}
DISCOURSE_RULES = {
    "D101": "仅从因为C所以E、因为/由于C，E的显式因果句中抽取完整原因C；问题所有内容词必须按真实词项顺序绑定结果E。",
    "D102": "仅从E是因为C、E是由于C、E的原因是C或E，因为C抽取原文原因C；不能把否定的因果关系改为肯定。",
    "D103": "从C导致/造成/致使E按原因到结果的明确方向抽取C；问题必须绑定E，不能反向回答C的原因。",
    "D104": "前句须完整且精确绑定所问结果，紧邻后句须以原因是/这是因为开头；保留两句真实来源和显式前件/连接表达，不跨章节、标题、列表或候选范围。",
    "D201": "X通过/采用/利用/使用M完成Y等句式只在明确施事及结果与问题逐词绑定时抽取M；方法与所问结果边界由真实词项及原文句式证明。",
    "D202": "X的方法是M或步骤是M只提供原文操作说明；X须完整绑定所问对象或事件，M保留原文，不声称该操作已经发生。",
    "D203": "要Y，可以通过/采用/利用/使用M及含明确可以/应当等措辞的方法句保留原文处方和情态，标为procedure；为了仅表示目的，不能作为事实原因。",
    "D204": "最多相邻两句的命名方法如下加首先/然后/最后步骤，仅在完整、无边界且前件精确绑定时引用整个步骤原句，不拼造跨句字段。",
    "DG101": "事实原因及事实方法拒绝引语、传闻、推测、计划、未来、假设及否定因果；原因内部的否定原样保留，结果的否定或取消须本身属于问题所问事件。",
    "DG102": "所有实质查询词、限定词、角色和已/曾/正在等状态须在正确的结果或方法角色中按真实词项顺序核验；不解析泛指代，不截取更长词项内部，不跨边界。",
    "DG103": "不同因果层、可选择的原因及不能唯一划定的方法边界不强选第一个；并列明确原因保留为一个完整原文区间。",
}
DISCOURSE_PATTERNS = {
    "cause_prefix": r"^(?:因为|由于)(?P<reason>[^，,；;。！？!?]+)[，,](?:(?:所以|因此|因而)[，,]?)?(?P<effect>[^，,；;。！？!?]+)$",
    "cause_predicate": r"^(?P<effect>[^，,；;。！？!?]+?)(?:的原因是|是因为|是由于)(?P<reason>[^，,；;。！？!?]+)$",
    "cause_postfix": r"^(?P<effect>[^，,；;。！？!?]+)[，,](?:因为|由于)(?P<reason>[^，,；;。！？!?]+)$",
    "cause_direction": r"^(?P<reason>[^，,；;。！？!?]+?)(?:导致|造成|致使)(?P<effect>[^，,；;。！？!?]+)$",
    "cause_reference": r"^(?P<reference>原因是|这是因为)(?P<reason>[^，,；;。！？!?]+)$",
    "cause_conditional": r"^(?:如果|假如|假设|倘若)(?:因为|由于)?(?P<reason>[^，,；;。！？!?]+)[，,](?:(?:那么|则|所以)[，,]?)?(?P<effect>[^，,；;。！？!?]+)$",
    "reported_prefix": r"^(?:据说|传闻|听说|据报道|有人声称)",
    "method_definition": r"^(?P<target>[^，,；;。！？!?]+?)(?:的方法是|的方式是|的步骤是)(?P<method>[^；;。！？!?]+)$",
    "method_marker": r"通过|采用|利用|使用",
    "method_instruction": r"^(?:若要|如需|要)(?P<effect>[^，,；;。！？!?]+)[，,](?P<modal>可以|可|应当|应该|应|需要|须|需)(?:通过|采用|利用|使用)(?P<method>[^；;。！？!?]+)$",
    "purpose_prefix": r"^为了(?P<purpose>[^，,；;。！？!?]+)[，,]",
    "method_modal": r"可以|能够|应当|应该|需要|必须|可|须|需",
    "method_mediation_token": r"^(?:要求|建议|安排|通知|让|使|被|由|帮助|协助|负责|因为|由于|导致|造成|致使)$",
    "nonfactual": r"计划(?=通过|采用|利用|使用|于|在|将|明年|未来|启动|开放|发射|开展|召开|完成|实施|建设|建造|取消|[0-9])|准备(?=通过|采用|利用|使用|于|在|将|启动|开放|发射|开展|完成|实施|建设|建造|取消|[0-9])|打算|原想|拟(?:于|将|通过|采用|利用|使用|启动|开放|发射|开展|完成|实施|建设|建造)|即将|将要|将会|将于|将(?=通过|采用|利用|使用|实现|完成|导致|造成|致使|启动|开放|降雨|[0-9])|可能|或许|也许|预计|预期|预定|有望|如果|假如|假设(?!检验|验证|分析)|倘若|据说|传闻|听说|声称|宣称|表示|认为|说道|说过|曾说|说[：:]|报道|[？?]",
    "quote": r"[“”‘’「」『』\"']",
    "negative_relation": r"不是因为|不是由于|并非因为|并非由于|不因|并非.+原因|不是.+原因|(?:并未|未|从未|不曾|没有|并不|不|不会|不能)(?:能|曾)?(?:直接|必然|真正|立即|明显|实际|最终|成功|有效)?(?:导致|造成|致使)|^(?:并不是|不是|并非).+(?:导致|造成|致使)",
    "effect_negative": r"没有|并未|尚未|未曾|未能|不会|不能|并非|不是|取消|停止|放弃|撤销|暂停|停运|未|没|不(?!超过|低于|小于|大于|足)",
    "alternative_cause": r"或者|或是|要么|还是|或(?!许)|不是.+就是",
    "reference": r"^(?:他|她|它|他们|她们|它们|其|这|那|该|此)$",
    "anchor_residue": r"(?:\s|的|了|已经|现已|已|曾经|曾|正式|成功|正在|在|于|地|得|计划|打算|准备|即将|将要|将会|将|会|要|可能|或许|也许|可以|能够|应当|应该|需要|必须|可|须|需|没有|并未|尚未|未曾|未能|不会|不能|并非|不是|并不|不|未|没|取消|停止|放弃|撤销|暂停|停运)*",
    "procedure_header": r"^(?P<target>[^，,；;。！？!?]+?)(?:的方法如下|的步骤如下|的操作步骤如下)[：:]?$",
    "procedure_steps": r"^首先[^；;。！？!?]+[，,](?:然后[^；;。！？!?]+[，,]){0,2}最后[^；;。！？!?]+$",
}


def _id(sentence):
    return sentence.get("sentence_id") or sentence["id"]


def _body(sentence):
    return sentence["text"].rstrip("。.!！")


def _field(index, sentence, start, end, rule):
    from .extraction import _span
    return _span(index, {**sentence, "sentence_id": _id(sentence)}, start, end, rule)


def _tokens(index, sentence):
    pids = set(sentence.get("paragraph_ids", []))
    seen, result = set(), []
    for paragraph in index["paragraphs"]:
        if paragraph["id"] not in pids:
            continue
        for token in paragraph.get("tokens", []):
            key = (token["clean_start"], token["clean_end"], token["term"])
            if (key not in seen and sentence["clean_start"] <= key[0]
                    and key[1] <= sentence["clean_end"]):
                seen.add(key)
                result.append(token)
    return sorted(result, key=lambda t: (t["clean_start"], t["clean_end"], t["term"]))


def _terms(result):
    # Keep repetitions: X affects X has two roles, not one bag-of-words hit.
    return [t["term"] for t in result["question"].get("retrieval_tokens", [])
            if t["term"] not in {"才", "才能"}]


def _anchor(index, result, zones):
    """Bind every ordered query token to genuine spans in semantic role zones.

    Extra source material is limited to explicitly listed grammatical/aspect
    markers.  Thus "甲的朋友帮助乙" cannot prove "甲帮助乙", even though all
    three query words occur in order.  A zone is (sentence, local start, end).
    """
    from .extraction import _alternatives
    terms = _terms(result)
    matches, zone_no, cursor = [], 0, zones[0][1] if zones else 0
    if not terms or not zones:
        return {"matched": False, "actual_term_spans": [], "query_terms": terms}
    occupied = [set() for _ in zones]
    for term in terms:
        forms, chosen = _alternatives(index, term), None
        for number in range(zone_no, len(zones)):
            sentence, start, end = zones[number]
            lower = max(start, cursor) if number == zone_no else start
            base = sentence["clean_start"]
            occurrences = [t for t in _tokens(index, sentence) if t["term"] in forms
                           and base + lower <= t["clean_start"] and t["clean_end"] <= base + end]
            # An expression match is usable only when ranking recorded a real
            # source occurrence, not merely a literal substring search.
            occurrences += [span for feature in sentence.get("features", [])
                            for match in feature.get("matches", []) if match["term"] in forms
                            for span in match.get("occurrences", [])
                            if base + lower <= span["clean_start"] and span["clean_end"] <= base + end]
            if occurrences:
                chosen = min(occurrences, key=lambda t: (t["clean_start"], t["clean_end"]))
                a, b = chosen["clean_start"] - base, chosen["clean_end"] - base
                occupied[number].update(range(a, b))
                matches.append({"query_term": term, "sentence_id": _id(sentence),
                                "clean_start": chosen["clean_start"], "clean_end": chosen["clean_end"]})
                zone_no, cursor = number, b
                break
        if chosen is None:
            return {"matched": False, "actual_term_spans": matches, "query_terms": terms,
                    "missing_term": term}
    residue = ["".join(sentence["text"][i] for i in range(a, b) if i not in used)
               for (sentence, a, b), used in zip(zones, occupied)]
    tight = all(re.fullmatch(DISCOURSE_PATTERNS["anchor_residue"], text) is not None for text in residue)
    return {"matched": tight, "actual_term_spans": matches, "query_terms": terms,
            "unmatched_role_text": residue, "all_ordered_terms_present": True}


def _unresolved(index, sentence, start, end):
    base = sentence["clean_start"]
    return [t["term"] for t in _tokens(index, sentence)
            if base + start <= t["clean_start"] and t["clean_end"] <= base + end
            and re.fullmatch(DISCOURSE_PATTERNS["reference"], t["term"])]


def _query_constraints(result, effect_text):
    problems = []
    for constraint in result["question"].get("constraints", []):
        kind, value = constraint.get("kind"), constraint.get("value")
        if kind in {"time", "location"}:
            # These remain literal query terms and must pass the role anchor.
            continue
        if kind == "status" and value == "modal":
            continue
        if kind == "status" and value in {"completed", "past", "ongoing"}:
            patterns = {"completed": r"已经|现已|已", "past": r"曾经|曾", "ongoing": r"正在"}
            if not re.search(patterns[value], effect_text):
                problems.append("unverified_query_" + value)
            continue
        problems.append("unsupported_query_" + str(kind) + "_" + str(value))
    if result.get("query_ambiguities"):
        problems.append("query_name_ambiguity")
    return problems


def _safety(index, result, sentences, effect_zones, reason_zone=None, procedure=False):
    issues = []
    if any(not sentence.get("complete") for sentence in sentences):
        issues.append("incomplete_sentence")
    text = "\n".join(sentence["text"] for sentence in sentences)
    if re.search(DISCOURSE_PATTERNS["quote"], text):
        issues.append("quotation")
    if re.search(DISCOURSE_PATTERNS["nonfactual"], text):
        issues.append("nonfactual_or_reported")
    if re.search(DISCOURSE_PATTERNS["negative_relation"], text):
        issues.append("negated_relation")
    effect_text = "".join(sentence["text"][a:b] for sentence, a, b in effect_zones)
    query_terms = _terms(result)
    for match in re.finditer(DISCOURSE_PATTERNS["effect_negative"], effect_text):
        # Cancellation/suspension can themselves be the explicitly asked event.
        # Negation filtered into a query constraint remains unsupported below.
        if not any(match.group() in term for term in query_terms):
            issues.append("effect_negative_or_cancelled")
    if not procedure and re.search(DISCOURSE_PATTERNS["method_modal"], effect_text):
        issues.append("source_modal_not_fact")
    if not procedure and any(t["term"] in {"将", "会", "要"}
                             for sentence in sentences for t in _tokens(index, sentence)):
        issues.append("source_future_or_intention")
    issues.extend(_query_constraints(result, effect_text))
    for sentence, a, b in effect_zones:
        if _unresolved(index, sentence, a, b):
            issues.append("unresolved_effect_reference")
    if reason_zone:
        sentence, a, b = reason_zone
        cause = sentence["text"][a:b]
        if re.search(DISCOURSE_PATTERNS["alternative_cause"], cause):
            issues.append("alternative_causes")
        if _unresolved(index, sentence, a, b):
            issues.append("unresolved_cause_reference")
        if re.search(DISCOURSE_PATTERNS["method_modal"], cause):
            issues.append("modal_cause")
    return sorted(set(issues))


def _method_issues(index, sentence, start, end):
    issues = []
    if end - start > DISCOURSE_CONFIG["max_method_chars"]:
        issues.append("method_scope_too_long")
    if _unresolved(index, sentence, start, end):
        issues.append("unresolved_method_reference")
    base = sentence["clean_start"]
    if any(re.fullmatch(DISCOURSE_PATTERNS["method_mediation_token"], t["term"])
           for t in _tokens(index, sentence)
           if base + start <= t["clean_start"] and t["clean_end"] <= base + end):
        issues.append("mediated_actor_or_causal_layer_in_method")
    return issues


def _generic_operation_query(result):
    """A query actor before 如何 cannot migrate to an object in the result."""
    question = result["question"]
    tokens = question.get("retrieval_tokens", [])
    markers = [m for m in question.get("type_matches", []) if m.get("type") == "method"]
    return bool(tokens and markers and tokens[0]["clean_start"] >= min(m["clean_end"] for m in markers))


def _decision(rule, sentences, checks, fields=None, statement_kind="fact", binding=None):
    issues = checks.get("unsafe_reasons", [])
    accepted = bool(fields and checks.get("matching_requested_roles") and not issues
                    and not checks.get("boundary_reasons"))
    code = rule if accepted else "DG103" if "alternative_causes" in issues else "DG101" if issues else "DG102"
    relevant = checks.get("matching_requested_roles") and not checks.get("boundary_reasons")
    blocking = ("ambiguous" if "alternative_causes" in issues else "unsafe") if relevant and issues else None
    return {"accepted": accepted, "rule": code, "matched_rule": rule,
            "reason": ("显式关系、真实词项角色和原文区间检查通过" if accepted else code + "：关系、角色、事实状态或边界检查未通过"),
            "sentence_ids": [_id(s) for s in sentences], "fields": fields if accepted else {},
            "statement_kind": statement_kind, "checks": checks, "blocking": blocking,
            "binding": binding or {"rule": rule, "checks": checks}}


def _reason_single(index, result, sentence):
    body = _body(sentence)
    reported = re.match(DISCOURSE_PATTERNS["reported_prefix"], body)
    prefix = reported.end() if reported else 0
    for key, rule in (("cause_prefix", "D101"), ("cause_predicate", "D102"),
                      ("cause_postfix", "D102"), ("cause_direction", "D103"),
                      ("cause_conditional", "D101")):
        match = re.fullmatch(DISCOURSE_PATTERNS[key], body[prefix:])
        if not match:
            continue
        spans = {name: (prefix + match.start(name), prefix + match.end(name)) for name in ("effect", "reason")}
        effect = (sentence, *spans["effect"])
        cause = (sentence, *spans["reason"])
        anchor = _anchor(index, result, [effect])
        checks = {"matching_requested_roles": anchor["matched"], "anchor": anchor,
                  "causal_direction": "reason_to_effect", "complete_sentences": bool(sentence.get("complete")),
                  "negative_cause_preserved": True,
                  "unsafe_reasons": _safety(index, result, [sentence], [effect], cause)}
        fields = {name: _field(index, sentence, *spans[name], rule) for name in ("reason", "effect")}
        return _decision(rule, [sentence], checks, fields)
    if re.search(r"因为|由于|原因是|导致|造成|致使|为了|相关", body):
        issues = _safety(index, result, [sentence], [(sentence, 0, len(body))])
        return _decision("D101", [sentence], {"matching_requested_roles": False,
                         "unsupported_relation_shape": True, "unsafe_reasons": issues})
    return None


def _method_single(index, result, sentence):
    body = _body(sentence)
    definition = re.fullmatch(DISCOURSE_PATTERNS["method_definition"], body)
    instruction = re.fullmatch(DISCOURSE_PATTERNS["method_instruction"], body)
    if definition or instruction:
        match, rule = (definition, "D202") if definition else (instruction, "D203")
        role = "target" if definition else "effect"
        zone = (sentence, *match.span(role))
        anchor = _anchor(index, result, [zone])
        checks = {"matching_requested_roles": anchor["matched"], "anchor": anchor,
                  "source_prescription": True,
                  "unsafe_reasons": _safety(index, result, [sentence], [zone], procedure=True)}
        checks["unsafe_reasons"].extend(_method_issues(index, sentence, *match.span("method")))
        fields = {"method": _field(index, sentence, *match.span("method"), rule),
                  role: _field(index, sentence, *match.span(role), rule),
                  "prescription": _field(index, sentence, 0, len(sentence["text"]), rule)}
        if instruction:
            fields["modality"] = _field(index, sentence, *match.span("modal"), rule)
        return _decision(rule, [sentence], checks, fields, "procedure")

    purpose = re.match(DISCOURSE_PATTERNS["purpose_prefix"], body)
    offset = purpose.end() if purpose else 0
    marker = re.search(DISCOURSE_PATTERNS["method_marker"], body[offset:])
    if not marker:
        return None
    marker_start, marker_end = offset + marker.start(), offset + marker.end()
    actor_end, actor_start = marker_start, offset
    modal = re.search("(?:" + DISCOURSE_PATTERNS["method_modal"] + ")$", body[actor_start:actor_end])
    if modal:
        actor_end = actor_start + modal.start()
    actor = (sentence, actor_start, actor_end)
    candidates = []
    # Only real indexed boundaries can divide the source method and result.
    boundaries = sorted({t["clean_start"] - sentence["clean_start"] for t in _tokens(index, sentence)
                         if marker_end < t["clean_start"] - sentence["clean_start"] < len(body)})
    for start in boundaries:
        method_end = start
        while method_end > marker_end and body[method_end - 1] in "，, ":
            method_end -= 1
        if method_end <= marker_end or any(p in body[marker_end:method_end] for p in "，,；;"):
            continue
        goal_start = start
        goal_modal = re.match(DISCOURSE_PATTERNS["method_modal"], body[goal_start:])
        if goal_modal:
            goal_start += goal_modal.end()
        goal = (sentence, goal_start, len(body))
        zones = ([actor] if actor_end > actor_start else []) + [goal]
        anchor = _anchor(index, result, zones)
        if not anchor["matched"] and _generic_operation_query(result):
            # A question naming only the operation may use a clearly described
            # source actor.  A named query actor still has to match the goal or
            # the explicit actor, in its actual grammatical order.
            anchor = _anchor(index, result, [goal])
            zones = [goal]
        if anchor["matched"]:
            candidates.append((start, method_end, goal, zones, anchor, goal_modal))
    if candidates:
        # Listed factual modifiers belong to the predicate, not the method.
        # Earliest matching complete predicate avoids stripping 成功/正式 from it.
        start, method_end, goal, zones, anchor, goal_modal = min(candidates, key=lambda row: row[0])
        procedure = bool(modal or goal_modal)
        rule = "D203" if procedure else "D201"
        checks = {"matching_requested_roles": True, "anchor": anchor,
                  "method_result_boundary": sentence["clean_start"] + start,
                  "method_marker": body[marker_start:marker_end],
                  "unsafe_reasons": _safety(index, result, [sentence], zones, procedure=procedure)}
        fields = {"method": _field(index, sentence, marker_end, method_end, rule),
                  "effect": _field(index, sentence, goal[1], goal[2], rule)}
        if actor_end > actor_start:
            fields["subject"] = _field(index, sentence, actor_start, actor_end, rule)
            if _unresolved(index, sentence, actor_start, actor_end):
                checks["unsafe_reasons"].append("unresolved_method_actor")
        if purpose:
            fields["purpose"] = _field(index, sentence, *purpose.span("purpose"), rule)
        if procedure:
            fields["prescription"] = _field(index, sentence, 0, len(sentence["text"]), rule)
            a = actor_start + modal.start() if modal else start
            b = marker_start if modal else start + goal_modal.end()
            fields["modality"] = _field(index, sentence, a, b, rule)
        checks["unsafe_reasons"].extend(_method_issues(index, sentence, marker_end, method_end))
        return _decision(rule, [sentence], checks, fields, "procedure" if procedure else "fact")

    # Purpose supplies an explicit operation target, but cannot prove an
    # achieved outcome.  Keep this as a procedure even without an extra modal.
    if purpose:
        zones = ([actor] if actor_end > actor_start else []) + [(sentence, *purpose.span("purpose"))]
        anchor = _anchor(index, result, zones)
        checks = {"matching_requested_roles": anchor["matched"], "anchor": anchor,
                  "purpose_not_achieved_effect": True,
                  "unsafe_reasons": _safety(index, result, [sentence], zones, procedure=True)}
        checks["unsafe_reasons"].extend(_method_issues(index, sentence, marker_end, len(body)))
        fields = {"method": _field(index, sentence, marker_end, len(body), "D203"),
                  "purpose": _field(index, sentence, *purpose.span("purpose"), "D203"),
                  "prescription": _field(index, sentence, 0, len(sentence["text"]), "D203")}
        if actor_end > actor_start:
            fields["subject"] = _field(index, sentence, actor_start, actor_end, "D203")
        return _decision("D203", [sentence], checks, fields, "procedure")
    return _decision("D201", [sentence], {"matching_requested_roles": False,
                     "unsupported_method_binding": True,
                     "unsafe_reasons": _safety(index, result, [sentence], [(sentence, 0, len(body))])})


def extract_discourse(index, result, evidence):
    """Return auditable accepted/rejected decisions within eligible sentences."""
    kind = result["question"].get("type")
    if kind not in {"reason", "method"}:
        return []
    eligible = {_id(s): s for s in evidence.get("candidates", [])}
    sentences = [{**s, **eligible[_id(s)], "sentence_id": _id(s)}
                 for s in index["sentences"]["sentences"] if _id(s) in eligible]
    decisions = []
    for sentence in sentences:
        decision = (_reason_single(index, result, sentence) if kind == "reason"
                    else _method_single(index, result, sentence))
        if decision:
            decisions.append(decision)

    from .joint_extraction import _blocked
    original = index["sentences"]["sentences"]
    for left, right in zip(original, original[1:]):
        if _id(left) not in eligible or _id(right) not in eligible:
            continue
        left, right = eligible[_id(left)], eligible[_id(right)]
        lbody, rbody = _body(left), _body(right)
        reference = re.fullmatch(DISCOURSE_PATTERNS["cause_reference"], rbody) if kind == "reason" else None
        header = re.fullmatch(DISCOURSE_PATTERNS["procedure_header"], lbody) if kind == "method" else None
        steps = bool(header and re.fullmatch(DISCOURSE_PATTERNS["procedure_steps"], rbody))
        if not reference and not steps:
            continue
        rule = "D104" if reference else "D204"
        zone = (left, 0, len(lbody)) if reference else (left, *header.span("target"))
        anchor = _anchor(index, result, [zone])
        checks = {"adjacent_in_document": True, "both_in_retrieval_scope": True,
                  "boundary_reasons": _blocked(index, left, right),
                  "matching_requested_roles": anchor["matched"], "anchor": anchor,
                  "single_complete_queried_antecedent": anchor["matched"] and not re.search(r"[，,；;]", lbody),
                  "unsafe_reasons": _safety(index, result, [left, right], [zone],
                                            (right, *reference.span("reason")) if reference else None,
                                            procedure=bool(steps))}
        checks["matching_requested_roles"] &= checks["single_complete_queried_antecedent"]
        fields = {"effect" if reference else "target": _field(index, left, zone[1], zone[2], rule),
                  "reason" if reference else "method": _field(index, right,
                         reference.start("reason") if reference else 0,
                         reference.end("reason") if reference else len(right["text"]), rule)}
        if steps:
            checks["unsafe_reasons"].extend(_method_issues(index, right, 0, len(right["text"])))
        if reference:
            fields["reference"] = _field(index, right, *reference.span("reference"), rule)
        binding = {"rule": rule, "antecedent": fields["effect" if reference else "target"],
                   "reference": fields.get("reference"), "checks": checks}
        decisions.append(_decision(rule, [left, right], checks, fields,
                                   "fact" if reference else "procedure", binding))
    return decisions
