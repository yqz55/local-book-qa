"""Deterministic sentence ranking, narrow span extraction and source-bound templates."""

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .ingest import SourceMap

POLICY = "general-source-answers/4"
CONFIG = {"window_limit": 20, "top_sentences": 3,
          "tie_break": "coverage descending, literal matches descending, original position ascending"}
RULES = {
    "F101": "从前 window_limit 个相邻句检索窗口取句子，按编号去重，并保留无单句关键词命中的邻句；不伪造跨行完整句。",
    "F102": "按查询计分组计算句内最大匹配权重；人物别名与原词归入同一概念。覆盖分=各概念最高匹配值之和/概念数。",
    "F103": "同分比较直接原词匹配数，再按原文位置和句号编号排序；匹配和分数不表示正确概率。",
    "X101": "定义只接受目标作为句首主语的 X是/X是指/X指/X定义为 句式；引用完整原句。",
    "X102": "时间只在同一分句中按真实词项边界和顺序绑定目标对象及事件，拒绝复杂施受事；只接受唯一日期或时间，不推断年份。",
    "X103": "数量只在同一分句中按真实词项边界和顺序绑定目标及属性，核对问题指定单位，保留单位、近似和范围；多个不明确数值不选第一个。",
    "X104": "赠送只接受 A把/将B送给/赠与/赠送给C 或 A赠送B给C 的显式角色句式；所有已知角色逐一核对。",
    "X105": "描述题只能引用完整原句，不能推导新结论。",
    "G101": "不完整句、复杂引语、否定、计划、假设、疑问、传闻或转述不能用于肯定事实短答。",
    "G102": "问题的否定/状态约束、未解决指代或人物歧义使事实短答停止；相关原文仍保留。",
    "G103": "同题候选有不同字段或一个分句有多个时间/数值时保留歧义；仅检查本次候选范围。",
    "G104": "字段必须来自同一原文区间，模板只填已有字段，缺字段不生成答案。",
}
PATTERNS = {
    "unsafe": r"没有|没|取消|停止|放弃|撤销|并未|尚未|未曾|未能|未送|不曾|不是|并非|不会|不能|不送|不赠|未|不(?!超过|低于|小于|大于|足)|无法|无意|计划|打算|准备|原想|(?:想|要|欲|愿|试图|尝试)(?:把|将|送|赠)|拟|将(?=[0-9]|于|会|要|在|送|赠|发射|启动|开始|结束|发生|发布|举办|召开)|可能|或许|也许|预计|预期|预定|有望|力争|应该|应当|如果|假如|假设|倘若|据说|传闻|听说|声称|询问|问道|表示|认为|宣称|说|[？?]",
    "quote": r"[“”‘’「」『』\"']",
    "complex_event": r"帮助|介绍|要求|建议|安排|通知|祝贺|负责|协助|让|被|把|向|对于|关于|而非|而是|新闻|报道|消息|后|前",
    "role_contamination": r"昨天|今天|明天|当时|随后|后来|已经|曾经|刚刚|于是|正在|然后|并且|离开|起身|说道|后|前|又|也|就",
    "date": r"(?:大约|约|大致)?(?:公元前|公元)?(?:(?:[0-9〇零一二三四五六七八九十百千万]{2,6}年)?[0-9]{1,2}月[0-9]{1,2}[日号](?:[0-9]{1,2}[时点][0-9]{1,2}分(?:[0-9]{1,2}秒)?)?|[0-9〇零一二三四五六七八九十百千万]{2,6}年(?:[0-9]{1,2}月)?|[0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2}|[0-9]{1,2}[时点][0-9]{1,2}分(?:[0-9]{1,2}秒)?)(?:前后|以前|之后|之前|以后|左右|年代|代|前|后)?",
    "quantity": r"(?:(?:不超过|不低于|不小于|不大于|大约|约|近|超过|至少|最多|不足|少于|大于|低于|高于)(?:为)?)?(?:零下|零上|[+−-]|负|正)?(?:[0-9]+(?:\.[0-9]+)?|[零〇一二三四五六七八九十百千万亿两]+)(?:\s*(?:至|到|～|~|—|–|-)\s*(?:[+−-]|负|正)?(?:[0-9]+(?:\.[0-9]+)?|[零〇一二三四五六七八九十百千万亿两]+))?\s*(?:平方千米|平方公里|平方米|立方米|千米/秒|千米|公里|厘米|毫米|公斤|千克|毫克|小时|分钟|个人|余人|万人|万元|亿元|摄氏度|毫升|升|℃|克|米|人|个|本|份|件|台|次|天|秒|元|%|％)(?:以上|以下|左右|以内|以外)?",
    "gift": r"(?P<subject>[^，,。！？!?；;：:\s]{1,16}?)(?:把|将)(?P<object>[^，,。！？!?；;：:]{1,24}?)(?P<verb>赠送给|送给|赠与|赠给|赠予)(?:了)?(?P<recipient>[^，,。！？!?；;：:]{1,16}?)(?:了)?[。！？!?；;]?$",
    "gift_alt": r"(?P<subject>[^，,。！？!?；;：:\s]{1,16}?)(?:赠送|赠与)(?P<object>[^，,。！？!?；;：:]{1,24}?)给(?P<recipient>[^，,。！？!?；;：:]{1,16}?)(?:了)?[。！？!?；;]?$",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def default_templates():
    return json.loads((Path(__file__).resolve().parents[1] / "data" / "answer_templates.json").read_text(encoding="utf-8"))


def answering_snapshot(snapshot=None):
    from .joint_extraction import JOINT_POLICY, JOINT_RULES
    from .evidence_groups import GROUP_POLICY, GROUP_CONFIG, GROUP_RULES
    from .discourse import DISCOURSE_POLICY, DISCOURSE_CONFIG, DISCOURSE_RULES, DISCOURSE_PATTERNS
    from .general_answers import general_snapshot
    if snapshot is None:
        from .questions import QUESTION_POLICY, QUESTION_RULES
        body = {"policy": POLICY, "config": CONFIG, "rules": RULES, "patterns": PATTERNS,
                "templates": default_templates(), "question_policy": QUESTION_POLICY, "question_rules": QUESTION_RULES,
                "joint_policy": JOINT_POLICY, "joint_rules": JOINT_RULES,
                "group_policy": GROUP_POLICY, "group_config": GROUP_CONFIG, "group_rules": GROUP_RULES,
                "discourse_policy": DISCOURSE_POLICY, "discourse_config": DISCOURSE_CONFIG,
                "discourse_rules": DISCOURSE_RULES, "discourse_patterns": DISCOURSE_PATTERNS,
                "general_answers": general_snapshot()}
    else:
        body = {k: v for k, v in snapshot.items() if k != "sha256"}
        # Templates/resources are embedded, but arbitrary executable rule changes
        # cannot silently impersonate this algorithm version.
        from .questions import QUESTION_POLICY, QUESTION_RULES
        if (body.get("policy") != POLICY or body.get("config") != CONFIG or body.get("rules") != RULES
                or body.get("patterns") != PATTERNS or body.get("question_policy") != QUESTION_POLICY
                or body.get("question_rules") != QUESTION_RULES or body.get("joint_policy") != JOINT_POLICY
                or body.get("joint_rules") != JOINT_RULES or body.get("group_policy") != GROUP_POLICY
                or body.get("group_config") != GROUP_CONFIG or body.get("group_rules") != GROUP_RULES
                or body.get("discourse_policy") != DISCOURSE_POLICY or body.get("discourse_config") != DISCOURSE_CONFIG
                or body.get("discourse_rules") != DISCOURSE_RULES or body.get("discourse_patterns") != DISCOURSE_PATTERNS
                or body.get("general_answers") != general_snapshot()):
            raise ValueError("不支持的问答规则快照。")
        if snapshot.get("sha256") != _digest(body):
            raise ValueError("问答规则快照哈希不一致。")
    required = {"definition": ["definition"], "time": ["time"], "quantity": ["value"],
                "relation": ["value"], "description": ["description"], "reason": ["reason"], "method": ["method"],
                "general": ["value"], "opinion": ["value"], "excerpt": ["description"]}
    if set(body["templates"]) != set(required):
        raise ValueError("答案模板类型不完整。")
    from string import Formatter
    for kind, fields in required.items():
        template = body["templates"][kind]
        if (set(template) != {"id", "required", "text"} or template["required"] != fields
                or not isinstance(template["id"], str) or not isinstance(template["text"], str)
                or [name for _, name, _, _ in Formatter().parse(template["text"]) if name is not None] != fields):
            raise ValueError("答案模板字段不符合来源约束。")
        if any(spec or conversion for _, name, spec, conversion in Formatter().parse(template["text"]) if name is not None):
            raise ValueError("答案模板只允许原样字段，不支持格式或转换表达式。")
    return deepcopy({**body, "sha256": _digest(body)})


def _groups(features):
    active = [f for f in features if not f.get("covered_by_expression")]
    parent = {f["term"]: f["term"] for f in active}
    def find(term):
        while parent[term] != term:
            parent[term] = parent[parent[term]]
            term = parent[term]
        return term
    score_groups = {}
    for f in active:
        group = f.get("score_group")
        if group in score_groups:
            parent[find(f["term"])] = find(score_groups[group])
        elif group:
            score_groups[group] = f["term"]
        for source in f["sources"]:
            original = source.get("from")
            if source["kind"] == "alias" and original in parent:
                parent[find(f["term"])] = find(original)
    groups = OrderedDict()
    for f in active:
        groups.setdefault(find(f["term"]), []).append(f)
    return list(groups.values())


def rank_sentences(index, result):
    cfg = index["answering"]["config"]
    selected = result["window_retrieval"]["candidates"][:cfg["window_limit"]]
    sentence_ids = {sid for window in selected for sid in window["sentence_ids"]}
    groups = _groups(result["query_terms"])
    occurrences = {}
    paragraphs = {p["id"]: p for p in index["paragraphs"]}
    # Explicit expression spans and word occurrences both keep actual boundaries.
    for group in groups:
        for feature in group:
            if feature["kind"] == "expression":
                spans = []
                for pid, positions in index["query_expansion"]["book_matches"].get(feature["term"], {}).items():
                    ts = paragraphs[pid]["tokens"]
                    spans.extend({"clean_start": ts[a]["clean_start"], "clean_end": ts[b - 1]["clean_end"],
                                  "source_start": ts[a]["source_start"], "source_end": ts[b - 1]["source_end"]} for a, b in positions)
            else:
                spans = [{k: t[k] for k in ("clean_start", "clean_end", "source_start", "source_end")}
                         for pid in index["postings"].get(feature["term"], []) for t in paragraphs[pid]["tokens"]
                         if t["term"] == feature["term"]]
            occurrences[feature["term"]] = spans
    candidates = []
    for sentence in index["sentences"]["sentences"]:
        if sentence["id"] not in sentence_ids:
            continue
        features, literals = [], set()
        for number, group in enumerate(groups, 1):
            matches = []
            for f in group:
                spans = [s for s in occurrences[f["term"]] if sentence["clean_start"] <= s["clean_start"]
                         and s["clean_end"] <= sentence["clean_end"]]
                if spans:
                    matches.append({"term": f["term"], "weight": f["weight"], "occurrences": spans,
                                    "sources": f["sources"]})
                    if f["query_frequency"] > 0:
                        literals.add(f["term"])
            value = max((m["weight"] for m in matches), default=0)
            features.append({"concept": number, "alternatives": [f["term"] for f in group], "matches": matches,
                             "match_value": value, "contribution": value / len(groups) if groups else 0})
        score = sum(f["contribution"] for f in features)
        candidates.append({**deepcopy(sentence), "sentence_id": sentence["id"], "score": score,
                           "features": features, "literal_matches": len(literals), "matched": score > 0,
                           "window_ids": [w["window_id"] for w in selected if sentence["id"] in w["sentence_ids"]]})
    candidates.sort(key=lambda s: (-s["score"], -s["literal_matches"], s["source_start"], s["sentence_id"]))
    for rank, s in enumerate(candidates, 1):
        s["rank"] = rank
    return {"policy": POLICY, "config": cfg, "rules": {k: RULES[k] for k in ("F101", "F102", "F103")},
            "searched_window_ids": [c["window_id"] for c in selected],
            "searched_paragraph_ids": list(dict.fromkeys(pid for c in selected for pid in c["paragraph_ids"])), "concept_count": len(groups),
            "candidate_count": len(candidates), "candidates": candidates,
            "top_ids": [s["sentence_id"] for s in candidates[:cfg["top_sentences"]]],
            "scope": "仅检查所列相邻句检索窗口包含的句子；无关键词邻句只提供候选上下文，不自动证明对象关联。不是全书无答案或一致性证明。"}


def _span(index, sentence, a, b, rule):
    mapping = SourceMap(index["cleaning"]["source_runs"], index["source"]["text"])
    ca, cb = sentence["clean_start"] + a, sentence["clean_start"] + b
    sa, sb = mapping.span(ca, cb)
    return {"text": index["source"]["text"][sa:sb], "source_start": sa, "source_end": sb,
            "clean_start": ca, "clean_end": cb, "sentence_id": sentence["sentence_id"], "rule": rule}


def _alternatives(index, value):
    values = {value}
    for group in index["aliases"]["groups"]:
        if value in group:
            values.update(group)
    return values


def _anchor_terms(result):
    # Literal query terms only; expanded senses cannot independently establish a
    # factual subject or a requested event.
    return list(dict.fromkeys(t["term"] for t in result["question"]["retrieval_tokens"]))


def _bound_clause(index, result, text, sentence, offset=0):
    terms = _anchor_terms(result)
    if not terms:
        return False
    # Ignore units asked as the unknown quantity, but retain event/property terms.
    units = {"克", "千克", "公斤", "米", "毫米", "公里", "个", "人", "元", "天", "年", "月", "日", "升", "毫升", "多少", "本", "份", "台", "次", "件", "小时", "分钟", "秒", "平方米", "平方公里", "立方米"}
    terms = [term for term in terms if term not in units]
    if not terms:
        return False
    event_text = re.sub(index["answering"]["patterns"]["date"], "", text)
    if re.search(index["answering"]["patterns"]["complex_event"], event_text):
        return False
    # Follow the actual query token order, using real indexed occurrences; a
    # character contained inside a different name/word is not an anchor match.
    start = sentence["clean_start"] + offset
    stop = start + len(text)
    cursor = start
    for term in terms:
        forms = _alternatives(index, term)
        spans = [span for feature in sentence["features"] for match in feature["matches"]
                 if match["term"] in forms for span in match["occurrences"]
                 if cursor <= span["clean_start"] and span["clean_end"] <= stop]
        # Covered fragments are matched inside their original contiguous phrase
        # only when index token boundaries prove those fragments.
        spans += [{"clean_start": t["clean_start"], "clean_end": t["clean_end"]}
                  for pid in sentence["paragraph_ids"] for t in index["paragraphs"][int(pid[1:]) - 1]["tokens"]
                  if t["term"] in forms and cursor <= t["clean_start"] and t["clean_end"] <= stop]
        if not spans:
            return False
        chosen = min(spans, key=lambda span: (span["clean_start"], span["clean_end"]))
        cursor = chosen["clean_end"]
    return True


def _requested_unit(result):
    units = ("平方千米", "平方公里", "平方米", "立方米", "千米/秒", "千米", "公里", "厘米", "毫米", "公斤", "千克", "毫克", "毫升", "小时", "分钟", "摄氏度", "克", "米", "人", "个", "本", "份", "件", "台", "次", "天", "秒", "元", "升", "%", "％", "℃")
    text = result["query_cleaning"]["text"]
    for marker in result["question"]["type_matches"]:
        if marker["type"] != "quantity":
            continue
        suffix = text[marker["clean_end"]:].lstrip()
        for unit in units:
            if suffix.startswith(unit) or marker["text"].endswith(unit):
                return unit
    return None


def _extract(index, result, sentence):
    kind, text = result["question"]["type"], sentence["text"]
    target = result["question"]["target"]
    patterns = index["answering"]["patterns"]
    if not sentence["complete"]:
        return None, "G101：句子不完整或引语边界不受支持"
    if kind == "description":
        if _bound_clause(index, result, text, sentence):
            return {"description": _span(index, sentence, 0, len(text), "X105")}, None
        return None, "目标内容未完整匹配"
    unsafe = re.search(patterns["unsafe"], text)
    if unsafe or re.search(patterns["quote"], text):
        return None, "G101：含否定、非事实状态、疑问或引语，保留原文核对"
    if kind == "definition":
        if not target:
            return None, "X101：未能确定被定义对象"
        for form in sorted(_alternatives(index, target), key=lambda x: (-len(x), x)):
            match = re.match(r"\s*" + re.escape(form) + r"(?:是指|定义为|指的是|是|指)(?P<body>.+?)[。.!！?？]?$", text)
            if match:
                return {"definition": _span(index, sentence, 0, len(text), "X101")}, None
        return None, "X101：未命中目标作为主语的定义句式"
    if kind in {"time", "quantity"}:
        values = []
        for clause in re.finditer(r"[^，,；;。！？!?]+", text):
            body = clause.group()
            # A leading date and subject/event may be separated by one comma.
            if kind == "time" and clause.start() == 0 and not _bound_clause(index, result, body, sentence, clause.start()):
                after = re.match(r"[，,]([^，,；;。！？!?]+)", text[clause.end():])
                if after and re.fullmatch(patterns["date"], body.strip()) and _bound_clause(index, result, after[1], sentence, clause.end() + 1):
                    values.append(_span(index, sentence, clause.start() + len(body) - len(body.lstrip()), clause.start() + len(body.rstrip()), "X102"))
                continue
            if not _bound_clause(index, result, body, sentence, clause.start()):
                continue
            pattern = patterns["date" if kind == "time" else "quantity"]
            matches = list(re.finditer(pattern, body))
            if len(matches) > 1:
                return None, "G103：同一分句有多个时间或数值，无法唯一绑定"
            for match in matches:
                # Numeric text in a date does not become a count.
                if kind == "quantity" and any(a.start() < match.end() and match.start() < a.end() for a in re.finditer(patterns["date"], body)):
                    continue
                if kind == "quantity" and _requested_unit(result):
                    unit = _requested_unit(result)
                    if not re.search(r"[0-9零〇一二三四五六七八九十百千万亿两]\s*" + re.escape(unit) + r"(?:以上|以下|左右|以内|以外)?$", match.group()):
                        continue
                values.append(_span(index, sentence, clause.start() + match.start(), clause.start() + match.end(), "X102" if kind == "time" else "X103"))
        if len(values) == 1:
            return {"time" if kind == "time" else "value": values[0]}, None
        return None, "G103：多个候选字段" if values else "未在目标对象及事件的同一受支持分句找到字段"
    if kind == "relation":
        relation = result["question"].get("relation")
        if not relation or relation.get("kind") != "gift" or not relation.get("requested_role"):
            return None, "X104：不支持该关系问法"
        match = re.fullmatch(patterns["gift"], text) or re.fullmatch(patterns["gift_alt"], text)
        if not match:
            return None, "X104：未命中显式赠送角色句式"
        fields = {}
        for role in ("subject", "object", "recipient"):
            value = match[role].removesuffix("了")
            if re.search(patterns["role_contamination"], value):
                return None, "X104：角色范围包含时间、副词或后续动作，无法可靠截取"
            if role in {"subject", "recipient"} and len(value) > 8:
                return None, "X104：人物角色不是受支持的短名称或称呼"
            known = relation.get(role)
            if known and known not in {"谁", "什么", "什么东西"} and value not in _alternatives(index, known):
                return None, "X104：已知角色不匹配：" + role
            if value in {"他", "她", "它", "他们", "她们", "此人", "其"}:
                return None, "G102：角色需要指代消解"
            fields[role] = _span(index, sentence, match.start(role), match.start(role) + len(value), "X104")
        fields["value"] = dict(fields[relation["requested_role"]])
        return fields, None
    return None, "未覆盖的抽取规则"


def answer_question(index, result, evidence):
    from .general_answers import answer_general, reference_response
    answer = _answer_specialized(index, result, evidence)
    if answer["status"] in {"answered", "ambiguous", "no_content"}:
        return {**answer, "mode": "extracted" if answer["status"] == "answered" else answer["status"]}
    general = answer_general(index, result, evidence, answer)
    return general if general is not None else reference_response(index, result, evidence, answer)


def _answer_specialized(index, result, evidence):
    question = result["question"]
    candidates = evidence["candidates"]
    base = {"policy": POLICY, "status": "insufficient_evidence", "text": "当前检索范围内未找到足够证据形成规则答案。",
            "template_id": None, "fields": {}, "evidence_ids": [], "reason": "证据或字段不足",
            "rule_ids": [], "decisions": [], "bindings": [], "scope": evidence["scope"]}
    if question["status"] == "no_content":
        return {**base, "status": "no_content", "text": "请补充要查询的对象、事件或术语。", "reason": "问题没有可用于定位的内容词"}
    kind = question["type"]
    if kind in {"reason", "method"}:
        return _answer_discourse(index, result, evidence, base)
    if kind == "unsupported":
        return {**base, "status": "unsupported", "text": "当前规则暂不支持该问题类型，以下提供相关原文。", "reason": "混合待答字段或开放式推理未被明确规则覆盖"}
    if kind == "keywords":
        return {**base, "status": "evidence_only" if candidates else "insufficient_evidence",
                "text": "以下是关键词对应的句子证据。" if candidates else base["text"], "reason": "关键词检索未指定抽取字段"}
    if (kind != "description" and (question.get("constraints") or result.get("query_ambiguities"))):
        return {**base, "status": "evidence_only" if candidates else "insufficient_evidence", "rule_ids": ["G102"],
                "text": "问题含事实约束或人物歧义，第一版保留原文供核对。", "reason": "G102：约束或人物歧义尚不能可靠核验"}
    accepted, decisions, unresolved, unsafe_relevant = [], [], False, False
    for sentence in candidates:
        fields, rejection = _extract(index, result, sentence)
        decision = {"sentence_id": sentence["sentence_id"], "accepted": fields is not None,
                    "fields": fields or {}, "reason": rejection or "字段与目标绑定，命中显式抽取规则"}
        decisions.append(decision)
        unresolved |= bool(rejection and rejection.startswith("G103"))
        unsafe_relevant |= bool(rejection and rejection.startswith("G101") and sentence["complete"]
                                and sentence["features"] and all(f["matches"] for f in sentence["features"]))
        if fields:
            accepted.append((sentence, fields, [sentence["sentence_id"]], None))
    from .joint_extraction import extract_joint
    joint_decisions = extract_joint(index, result, evidence)
    decisions.extend(joint_decisions)
    by_id = {s["sentence_id"]: s for s in candidates}
    for decision in joint_decisions:
        unresolved |= decision.get("rule") == "JG103"
        unsafe_relevant |= decision.get("rule") == "JG101"
        if decision["accepted"]:
            ids = decision["sentence_ids"]
            accepted.append((by_id[ids[-1]], decision["fields"], ids, decision["binding"]))
    if not accepted:
        return {**base, "status": "ambiguous" if unresolved else "evidence_only" if candidates else "insufficient_evidence",
                "text": "候选字段存在歧义，请核对原文。" if unresolved else "找到相关原文，但未满足规则抽取条件。" if candidates else base["text"],
                "reason": "G103：字段无法唯一绑定" if unresolved else base["reason"], "decisions": decisions}
    template = index["answering"]["templates"][kind]
    # Complete definitions may have different explanatory wording. Surface both
    # rather than pretending to prove that their meanings are equivalent.
    signatures = {tuple(fields[f]["text"] for f in template["required"]) for _, fields, _, _ in accepted}
    if kind != "description" and (len(signatures) > 1 or unresolved or unsafe_relevant):
        return {**base, "status": "ambiguous", "text": "候选证据给出不同或无法唯一绑定的字段，请核对原文。",
                "reason": "G103：本次候选范围内存在不同字段", "rule_ids": ["G103"], "decisions": decisions,
                "evidence_ids": list(dict.fromkeys(sid for _, _, ids, _ in accepted for sid in ids))}
    sentence, fields, ids, binding = accepted[0]
    if any(f not in fields for f in template["required"]):
        return {**base, "reason": "G104：模板字段缺失", "rule_ids": ["G104"], "decisions": decisions}
    return {**base, "status": "answered", "text": template["text"].format(**{k: v["text"] for k, v in fields.items()}),
            "template_id": template["id"], "fields": fields, "evidence_ids": ids, "bindings": [binding] if binding else [],
            "reason": "全部必需字段来自原文，适用条件通过", "rule_ids": sorted({v["rule"] for v in fields.values()}) + ["G104"],
            "template": template, "decisions": decisions}


def _answer_discourse(index, result, evidence, base):
    """Organize only validated source-stated causes or methods.

    Different source explanations are kept as alternatives. They are not
    silently joined into a novel causal claim or asserted to contradict.
    """
    from .discourse import extract_discourse
    kind = result["question"]["type"]
    label = "原因" if kind == "reason" else "方法"
    decisions = extract_discourse(index, result, evidence)
    accepted = [decision for decision in decisions if decision.get("accepted")]
    blockers = [decision for decision in decisions if decision.get("blocking") in {"unsafe", "ambiguous"}]
    if not accepted:
        ambiguous = any(decision.get("blocking") == "ambiguous" for decision in blockers)
        return {**base, "status": "ambiguous" if ambiguous else "evidence_only" if evidence["candidates"] else "insufficient_evidence",
                "text": f"候选{label}无法唯一绑定，请核对原文。" if ambiguous else
                        f"找到相关原文，但未找到与问题明确绑定的{label}表述。" if evidence["candidates"] else base["text"],
                "reason": f"{label}字段不足、关系未成立或原文状态未满足规则条件", "decisions": decisions,
                "rule_ids": list(dict.fromkeys(d["rule"] for d in decisions))}
    template = index["answering"]["templates"][kind]
    for decision in accepted:
        if any(field not in decision["fields"] for field in template["required"]):
            raise ValueError("原因/方法抽取缺少模板必需字段。")
    signatures = {(decision.get("statement_kind", "fact"),
                   *(decision["fields"][name]["text"] for name in template["required"])) for decision in accepted}
    ids = list(dict.fromkeys(sid for decision in accepted for sid in decision["sentence_ids"]))
    if len(signatures) > 1 or blockers:
        return {**base, "status": "ambiguous", "text": f"原文存在不同或尚未确认的{label}候选，请分别核对证据。",
                "reason": "不同表述可能是补充、替代或受到限制；当前规则不擅自合并或判定矛盾。",
                "evidence_ids": ids, "decisions": decisions,
                "rule_ids": list(dict.fromkeys(d["rule"] for d in accepted + blockers))}
    selected = accepted[0]
    fields = selected["fields"]
    return {**base, "status": "answered", "text": template["text"].format(**{key: value["text"] for key, value in fields.items()}),
            "template_id": template["id"], "template": deepcopy(template), "fields": deepcopy(fields),
            "evidence_ids": list(selected["sentence_ids"]), "bindings": [deepcopy(selected["binding"])] if selected.get("binding") else [],
            "statement_kind": selected.get("statement_kind", "fact"),
            "reason": f"原文明示{label}关系，目标和字段来源检查通过",
            "rule_ids": [selected["rule"], "G104"], "decisions": decisions}
