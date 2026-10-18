"""Deterministic question plans, retaining the original tokenizer coordinates.

This module does not rewrite a query into invented words.  It selects existing
tokens, records why other tokens were excluded, and keeps grammatical facts in
an inspectable plan for the sentence/extraction stages.
"""

import re

from .ingest import SourceMap


QUESTION_POLICY = "question-plan/1"
QUESTION_RULES = {
    "QP001": "先按最长完整疑问表达识别题型；疑问表达及礼貌前缀不参与计分或扩展。",
    "QP002": "保留已发现姓名、本书专用词、书名号及引号内的字面检索对象；不删除词项内部的子串。",
    "QP003": "仅在识别为问句时排除独立语法词及句首文档语境表达；检索词始终为原始分词结果的子集，不重写词项或坐标。",
    "QP004": "问句中的否定及计划、假设等事实状态记录为约束并排除普通词项计分；明确时间、地点约束仍可检索；未解决代词单独记录。",
    "QP005": "赠送关系仅接受列出的显式句式，记录施事、物品、受事及待回答角色；不推断代词或省略。",
    "QP006": "剔除提问表达后没有实质内容时返回 no_content；纯关键词输入不作全局停用词删除；多个待回答类型不强选一个字段。",
}

# Longest expressions win at an identical position.  In particular, the 什么
# inside 为什么/什么时候 must never become a second question slot.
_EXPRESSIONS = {
    "reason": ("出于什么原因", "出于何种原因", "是什么原因", "什么原因", "是为什么", "是为何", "为什么", "为何", "何故"),
    "description": ("是怎样的", "是什么样的", "如何描述", "怎样描述", "怎么描述", "如何介绍", "怎样介绍"),
    "time": ("是在什么时候", "是什么时候", "是什么时间", "在什么时候", "什么时候", "什么时间", "是何时", "哪一年", "哪一月", "哪一天", "何时", "哪年", "哪月", "哪天", "几点", "几时"),
    "method": ("通过什么方式", "以什么方式", "用什么方法", "是怎样", "是如何", "是怎么", "怎样", "如何", "怎么"),
    "quantity": ("有多少", "是多少", "多少", "几个", "几名", "几位", "几次", "几天", "几年", "多大", "多长", "多高", "多重"),
    "definition": ("什么叫做", "什么是", "是什么", "指什么", "什么叫", "何谓"),
    "relation": ("谁",),
    "unsupported": ("出现在哪里", "在哪里", "在何处", "哪里", "何处", "哪种", "哪个", "哪些", "什么"),
}
_PREFIXES = ("请问一下", "请告诉我", "能否告诉我", "能告诉我", "我想知道", "请解释一下", "请介绍一下", "请问", "告诉我", "想知道")
_PARTICLES = frozenset(("的", "了", "吗", "呢", "呀", "吧", "啊", "是", "把", "将", "给", "于", "在", "请", "一下"))
_QUANTITY_PARTICLES = frozenset(("有", "共有", "拥有", "一共有", "总共", "一共", "共"))
_AUXILIARIES = ("可以", "能够", "应该", "应当", "要", "会", "能")
_DOCUMENT_CONTEXT = ("文章中", "文献中", "书中", "文中", "本文", "书里", "文章里")
_CONTEXT_ONLY = frozenset(("这", "那", "这个", "那个", "这些", "那些", "这样", "那样", "如此", "它", "他们", "她们", "它们",
                           "会", "能", "可以", "能够", "应该", "应当", "要", "需要", "做", "发生"))
_REFERENCES = frozenset(("他", "她", "它", "他们", "她们", "它们", "此人", "其"))
_NEGATIONS = ("并没有", "并不是", "没有", "并未", "尚未", "从未", "未曾", "不曾", "不是", "不能", "不会", "不", "没", "未")
_STATES = {
    "planned": ("原本打算", "原想", "打算", "计划", "准备", "拟于", "拟将", "拟"),
    "hypothetical": ("假如", "如果", "假设", "倘若"),
    "uncertain": ("是否", "能否", "可能", "也许", "或许"),
    "future": ("将要", "即将", "将于", "将会"),
    "past": ("曾经", "曾"),
    "completed": ("已经", "已"),
    "ongoing": ("正在",),
}
_TIME = re.compile(r"(?:公元前)?(?:\d{2,4}|[〇零一二三四五六七八九十百千]+)年(?:\d{1,2}月)?(?:\d{1,2}[日号])?|\d{1,2}月\d{1,2}[日号]|\d{1,2}(?:[:：]\d{2}|点(?:\d{1,2}分)?)")
_LOCATION = re.compile(r"(?:位于|在)(?P<place>[^\s，。！？?！、；;的]{1,12}?)(?=于|由|把|将|进行|举办|举行|成立|建立|启动|发布|发射|出生|工作|学习|召开|建造|赠送|送给|赠给|赠予)")
_QUOTES = re.compile(r"《[^》\n]+》|“[^”\n]+”|‘[^’\n]+’|\"[^\"\n]+\"|'[^'\n]+'")


def _overlaps(a, b, start, end):
    return a < end and start < b


def analyze_question(cleaned: dict, raw: str, tokens: list[dict]) -> dict:
    """Return a replayable plan; ``retrieval_tokens`` contains original tokens.

    All intervals are half-open Unicode-character positions, identical to
    ingestion/tokenization.  ``constraints`` values are observations about the
    question, not assertions that the corresponding source event happened.
    """
    text = cleaned["text"]
    mapping = SourceMap(cleaned["source_runs"], raw)
    token_spans = [(t["clean_start"], t["clean_end"]) for t in tokens]
    protected = [(t["clean_start"], t["clean_end"]) for t in tokens if t.get("rule") in {"N001", "D005"}]
    protected += [m.span() for m in _QUOTES.finditer(text)]

    def item(start, end, **extra):
        source_start, source_end = mapping.span(start, end)
        return {"text": text[start:end], "clean_start": start, "clean_end": end,
                "source_start": source_start, "source_end": source_end, **extra}

    def is_protected(start, end):
        return any(_overlaps(a, b, start, end) for a, b in protected)

    def complete(start, end):
        # A raw-text match may span many small tokens, but may not excise a
        # substring of an existing dictionary/name/book token.
        return not any(_overlaps(a, b, start, end) and not (start <= a and b <= end)
                       for a, b in token_spans)

    def match_spans(options, category):
        found = []
        for value, values in options.items():
            for expression in values:
                for m in re.finditer(re.escape(expression), text):
                    start, end = m.span()
                    if not is_protected(start, end) and complete(start, end):
                        found.append(item(start, end, category=category, value=value))
        found.sort(key=lambda row: (row["clean_start"], -(row["clean_end"] - row["clean_start"]), row["value"]))
        accepted = []
        for row in found:
            if not any(_overlaps(a["clean_start"], a["clean_end"], row["clean_start"], row["clean_end"]) for a in accepted):
                accepted.append(row)
        return accepted

    type_matches = match_spans(_EXPRESSIONS, "question_expression")
    # Polite requests are recognized at the beginning, independently of how the
    # dictionary tokenizes them.  They are never stripped inside an object name.
    polite = []
    for expression in sorted(_PREFIXES, key=lambda s: (-len(s), s)):
        for m in re.finditer(re.escape(expression), text):
            if text[:m.start()].strip(" \t\n，,：:"):
                continue
            if complete(*m.span()) and not is_protected(*m.span()) and not polite:
                polite.append(item(*m.span(), category="polite_prefix", value=expression))
    # A trailing '?' is also common on a keyword search (“送给？”).  Require
    # actual interrogative grammar before filtering its ordinary word tokens.
    question_context = bool(type_matches or polite or (tokens and tokens[-1]["term"] in {"吗", "呢"}))
    priority = ("reason", "description", "time", "method", "quantity", "relation", "definition", "unsupported")
    question_type = next((kind for kind in priority if any(m["value"] == kind for m in type_matches)), "keywords")

    negations = match_spans({"negative": _NEGATIONS}, "negation") if question_context else []
    states = match_spans(_STATES, "status") if question_context else []
    document_context = []
    if question_context:
        states += match_spans({"modal": _AUXILIARIES}, "status")
        for context in match_spans({"context": _DOCUMENT_CONTEXT}, "document_context"):
            # Context removal is only a prefix rule.  A reference to 本文 in
            # the middle of a queried object must not be silently excised.
            preceding = list(text[:context["clean_start"]])
            for prefix in polite:
                if prefix["clean_end"] <= context["clean_start"]:
                    preceding[prefix["clean_start"]:prefix["clean_end"]] = [" "] * (prefix["clean_end"] - prefix["clean_start"])
            if not "".join(preceding).strip(" \t\n，,：:"):
                document_context.append(context)
    # A status word contained in an already identified question/polite expression
    # (e.g. 能否告诉我) is grammatical, not an event-status constraint.
    grammatical = type_matches + polite + document_context
    def outside_grammar(row):
        return not any(_overlaps(row["clean_start"], row["clean_end"], x["clean_start"], x["clean_end"]) for x in grammatical)
    negations = [x for x in negations if outside_grammar(x)]
    states = [x for x in states if outside_grammar(x)]
    constraints = [{**x, "kind": x["category"]} for x in negations + states]
    for m in _TIME.finditer(text):
        constraint = item(*m.span(), category="time", kind="time", value=m.group())
        if not is_protected(*m.span()) and outside_grammar(constraint):
            constraints.append(constraint)
    for m in _LOCATION.finditer(text):
        constraint = item(*m.span("place"), category="location", kind="location", value=m.group("place"))
        if not is_protected(*m.span("place")) and outside_grammar(constraint):
            constraints.append(constraint)
    if question_context:
        for token in tokens:
            if token["term"] in _REFERENCES and not is_protected(token["clean_start"], token["clean_end"]):
                constraints.append(item(token["clean_start"], token["clean_end"], category="reference", kind="reference", value="unresolved"))
    constraints.sort(key=lambda x: (x["clean_start"], x["clean_end"], x["kind"]))

    removed = [*type_matches, *polite, *document_context, *negations, *states]
    decisions = []
    for row in removed:
        rule = "QP004" if row["category"] in {"negation", "status"} else "QP003" if row["category"] == "document_context" else "QP001"
        decisions.append({**row, "action": "filtered", "rule": rule,
                          "reason": "用于事实约束，不参与普通内容计分" if rule == "QP004" else "句首文档语境，不参与内容计分" if rule == "QP003" else "用于提问意图或礼貌请求，不参与内容计分"})
    retrieval_tokens = []
    for token in tokens:
        start, end = token["clean_start"], token["clean_end"]
        excluded = next((r for r in removed if _overlaps(start, end, r["clean_start"], r["clean_end"])), None)
        if excluded:
            continue
        term = token["term"]
        # Pure keyword queries retain their existing behavior.  Even in a
        # question, only whole independent grammar tokens are removed.
        particles = _PARTICLES | (_QUANTITY_PARTICLES if question_type == "quantity" else frozenset())
        if question_context and term in particles and not is_protected(start, end):
            row = item(start, end, category="particle", value=term)
            removed.append(row)
            decisions.append({**row, "action": "filtered", "rule": "QP003", "reason": "问句中的独立语法词"})
            continue
        retrieval_tokens.append(token)
        protected_token = is_protected(start, end)
        decisions.append(item(start, end, category="content", value=term, action="retained",
                              rule="QP002" if protected_token else "QP003",
                              reason="保留姓名、专用词或引号内字面对象" if protected_token else "保留原始内容词及其坐标"))

    # “为什么会这样” does not name an object or event.  Restrict this rule to
    # all-context queries: 会 in “水为什么会蒸发” must not erase its real anchor.
    if question_context and retrieval_tokens and all(t["term"] in _CONTEXT_ONLY and not is_protected(t["clean_start"], t["clean_end"])
                                                     for t in retrieval_tokens):
        for token in retrieval_tokens:
            start, end = token["clean_start"], token["clean_end"]
            decisions = [d for d in decisions if not (d["clean_start"] == start and d["clean_end"] == end and d["action"] == "retained")]
            row = item(start, end, category="missing_context", value=token["term"])
            removed.append(row)
            decisions.append({**row, "action": "filtered", "rule": "QP006", "reason": "仅有指代或助动表达，缺少可定位的对象或事件"})
        retrieval_tokens = []

    relation = _gift_relation(text, tokens, mapping, type_matches, polite + document_context, negations + states)
    if relation and relation["requested_role"] and question_type in {"unsupported", "definition", "relation", "keywords"}:
        question_type = "relation"
    if relation and relation["question_role_count"] > 1:
        question_type = "unsupported"
    question_types = sorted({m["value"] for m in type_matches})
    if len(question_types) > 1:
        question_type = "unsupported"
    target = _target(text, retrieval_tokens, removed, question_type)
    keywords = list(dict.fromkeys(t["term"] for t in retrieval_tokens))
    # Generic single-word questions such as “什么？” have no lexical anchor.
    status = "ok" if retrieval_tokens else "no_content"
    return {"policy": QUESTION_POLICY, "rules": QUESTION_RULES, "type": question_type,
            "type_matches": [{**m, "type": m["value"], "rule": "QP001"} for m in type_matches],
            "status": status, "target": target, "target_terms": keywords,
            "question_types": question_types,
            "relation": relation, "constraints": constraints,
            "decisions": sorted(decisions, key=lambda x: (x["clean_start"], x["clean_end"], x["action"])),
            "retrieval_tokens": retrieval_tokens, "keywords": keywords}


def _target(text, tokens, removed, question_type):
    """A conservative, literal anchor, never a generated paraphrase."""
    keep = [True] * len(text)
    for row in removed:
        keep[row["clean_start"]:row["clean_end"]] = [False] * (row["clean_end"] - row["clean_start"])
    target = "".join(char for i, char in enumerate(text) if keep[i] and (char.isalnum() or "\u3400" <= char <= "\u9fff"))
    if question_type != "keywords":
        target = re.sub(r"^(?:(?:请|在)?(?:本文|文章|文中|书中|文内|书里|文章中|文献中)|关于|对于|针对)+", "", target)
        if question_type == "definition":
            target = re.sub(r"(?:的定义|定义|的含义|含义|的意思|意思)$", "", target)
    return target


_GIFT_VERBS = r"赠送给|赠给|送给|赠予|赠与"
_GIFT_PATTERNS = (
    ("QR001", re.compile(rf"(?P<subject>.{{1,40}}?)(?:把|将)(?P<object>.{{1,60}}?)(?P<verb>{_GIFT_VERBS})(?:了)?(?P<recipient>.{{1,40}})")),
    ("QR002", re.compile(r"(?P<subject>.{1,40}?)给(?P<recipient>.{1,40}?)(?P<verb>赠送|送)(?:了)?(?P<object>.{1,60})")),
    ("QR003", re.compile(r"(?P<subject>.{1,40}?)(?P<verb>赠送|送)(?:了)?(?P<object>.{1,60}?)给(?P<recipient>.{1,40})")),
)


def _gift_relation(text, tokens, mapping, questions, polite, constraints):
    # Remove only non-role questions and state markers; compacting with a map
    # lets the role records continue to cite the original question coordinates.
    skip = [*polite, *constraints, *(q for q in questions if q["text"] not in {"谁", "什么"})]
    positions = [i for i, char in enumerate(text) if not char.isspace() and char not in "，,。！？?!：:；;"
                 and not any(r["clean_start"] <= i < r["clean_end"] for r in skip)]
    compact = "".join(text[i] for i in positions)
    compact = compact.rstrip("吗呢呀吧啊")
    positions = positions[:len(compact)]
    matched, rule = None, None
    for candidate_rule, pattern in _GIFT_PATTERNS:
        matched = pattern.fullmatch(compact)
        if matched:
            rule = candidate_rule
            break
    if not matched:
        # Recipient-first direct gifts need an explicit 谁, a known name, or a
        # visible numeral/classifier boundary to separate recipient and object.
        direct = re.fullmatch(rf"(?P<subject>.{{1,40}}?)(?P<verb>{_GIFT_VERBS})(?:了)?(?P<tail>.{{2,80}})", compact)
        if not direct:
            return None
        tail, offset = direct.group("tail"), direct.start("tail")
        split = 1 if tail.startswith("谁") else None
        if split is None:
            for token in tokens:
                if token.get("rule") == "N001" and token["clean_start"] == positions[offset]:
                    split = len(token["surface"])
                    break
        if split is None:
            numeral = re.search(r"(?:[一二三四五六七八九十两百千万零〇]+|\d+)(?:个|件|块|本|把|支|只|张|封|台|份|套|辆|枚|朵|束|杯|条)", tail)
            if numeral and numeral.start() > 0:
                split = numeral.start()
        if split is None or split >= len(tail):
            return None
        spans = {"subject": direct.span("subject"), "recipient": (offset, offset + split),
                 "object": (offset + split, direct.end("tail")), "verb": direct.span("verb")}
        rule = "QR004"
    else:
        spans = {name: matched.span(name) for name in ("subject", "object", "recipient", "verb")}

    def original_span(start, end):
        a, b = positions[start], positions[end - 1] + 1
        source_start, source_end = mapping.span(a, b)
        return {"text": text[a:b], "clean_start": a, "clean_end": b,
                "source_start": source_start, "source_end": source_end}

    roles = {role: original_span(*spans[role]) for role in ("subject", "object", "recipient")}
    requested = []
    values = {}
    for role, record in roles.items():
        value = compact[spans[role][0]:spans[role][1]]
        unknown = value in {"谁", "什么", "什么东西", "何人"}
        record["is_question"] = unknown
        values[role] = None if unknown else value
        if unknown:
            requested.append(role)
    return {"kind": "gift", "rule": rule, "verb": compact[spans["verb"][0]:spans["verb"][1]],
            "verb_span": original_span(*spans["verb"]), **values, "roles": roles,
            "requested_role": requested[0] if len(requested) == 1 else None,
            "question_role_count": len(requested)}
