"""Literal, reusable answer-slot plans; no generated search terms or models."""

import re


GENERAL_QUESTION_POLICY = "literal-answer-slots/2"
GENERAL_QUESTION_RULES = {
    "QP101": "疑问表达只确定待答字段；身份、属性、行为、地点、结果与原文描述另存结构，不把所有谁限定为赠送。",
    "QP102": "所有对象、关系和槽位来自问题原文；作者/笔者的评价归属单列，不用它们代替正文中的人物。",
    "QP103": "开放问题、列举、概括、判断和复合问题均进入证据回答流程；未能证明字段时使用明确标记的原文参考回答。",
    "QP104": "自然量词问法保留对象类别；材料、明确事件地点及方向端点分别建立结构字段，不把类别词或修饰语任意删除。",
}
QUESTION_PATTERNS = {
    "author": r"^(?:作者|笔者|叙述者)(?:认为|觉得|感到|指出|说|提到|眼中|看来)[，,：:]?",
    "identity_subject": r"^(?:谁|什么人)(?:才)?是(?P<predicate>.+)$",
    "identity_value": r"^(?P<subject>.+?)(?:是(?:谁|何人|什么人)|叫什么(?:名字)?|叫啥)$",
    "passive_actor": r"^(?P<object>.+?)(?:是由|是被|由|被|是)谁(?P<predicate>.+?)(?:的)?$",
    "action": r"^(?P<subject>.+?)(?:做了|做过|干了|进行了|从事)(?:什么|哪些)(?:事情|事|工作|活动)?$",
    "state_description": r"^(?P<subject>.+?)(?P<predicate>保持着|保持|保留着|保留|呈现|具有|拥有|处于)(?:怎样的|什么样的|什么样)(?P<property>.+)$",
    "attribute": r"^(?P<subject>.+?)的(?P<property>[^的]{1,20}?)(?:是(?:什么|哪些)|有(?:哪些|什么)|为(?:什么))$",
    "typed_attribute": r"^(?P<subject>.+?)(?:是(?:什么|哪种))(?P<property>颜色|形状|材质|材料|类型|职业|身份|关系)$",
    "material": r"^(?P<subject>.+?)(?:是)?(?:用|由)(?:什么|哪种)(?:材料|材质)(?P<predicate>砌成|制成|建成|做成|制造)(?:的)?$",
    "consequence": r"^(?P<subject>.+?)(?:的)?(?P<priority>最重要|重要|主要|最大)?(?:的)?(?P<property>贡献|作用|影响|结果|成就|意义|功劳)(?:是(?:什么|哪些)|有(?:哪些|什么))$",
    "location": r"^(?P<subject>.+?)(?:位于|坐落于|设在|在)(?:哪里|何处|什么地方)(?P<suffix>.*)$",
    "event_location": r"^(?P<subject>.+?)(?P<predicate>准确着陆|着陆|降落|落地|抵达|到达)(?:在|于|到)?(?:哪里|何处|什么地方)$",
    "endpoints": r"^(?P<subject>.+?)的(?P<first>北|南|东|西)端和(?P<second>北|南|东|西)端分别是(?:什么|哪里)$",
    "description": r"怎样的|什么样的|如何描述|怎样描述|怎么描述|如何介绍|怎样介绍",
    "summary": r"概括|总结|主要内容|讲了什么|讲述了什么|说明了什么|介绍了什么|中心思想|主题|启示|如何评价|怎么看",
    "source_context": r"^(?:(?:课文|文中|文章|本文|原文)(?:中)?(?:说|指出|提到|列出|列出的|介绍|记载|说明|显示)?[，,：:]?)+",
    "slot": r"谁|什么(?:东西|材料|材质|地方|地点|人)?|哪(?:一|两|二|三|四|五|六|七|八|九|十|几)?(?:个|位|种|类|本|条|座|处|项|些)(?:期刊|杂志|刊物|材料|材质|地方|地点|人|东西)?",
}


def build_general_plan(question, cleaned, raw):
    """A plan supplements the historical parser without rewriting its tokens."""
    text = cleaned["text"]
    # Coordinates in this plan refer to the unchanged clean question. Source
    # coordinates on actual question tokens remain owned by analyze_question.
    start, stop = 0, len(text)
    while stop > start and text[stop - 1] in " \t\n。！？?!吗呢呀吧啊":
        stop -= 1
    for row in question["decisions"]:
        if row.get("category") not in {"polite_prefix", "document_context"}:
            continue
        if not text[start:row["clean_start"]].strip(" \t\n，,：:"):
            start = row["clean_end"]
    while start < stop and text[start] in " \t\n，,：:":
        start += 1
    body = text[start:stop]
    source_context = re.match(QUESTION_PATTERNS["source_context"], body)
    if source_context:
        start += source_context.end()
        body = text[start:stop]
    author = re.match(QUESTION_PATTERNS["author"], body)
    context = None
    if author:
        context = {"kind": "author", "text": author.group(), "clean_start": start,
                   "clean_end": start + author.end()}
        start += author.end()
        body = text[start:stop]
    plan = {"policy": GENERAL_QUESTION_POLICY, "intent": question["type"], "body": body,
            "body_clean_start": start, "attribution": context, "slots": [], "subject": None,
            "predicate": None, "property": None, "suffix": "", "rule_ids": ["QP101"]}
    for name, intent in (("identity_subject", "identity"), ("identity_value", "identity"),
                         ("passive_actor", "event_actor"), ("action", "action"),
                         ("state_description", "state_description"),
                         ("consequence", "consequence"), ("endpoints", "endpoints"),
                         ("attribute", "attribute"),
                         ("typed_attribute", "attribute"), ("material", "material"),
                         ("event_location", "event_location"),
                         ("location", "location")):
        match = re.fullmatch(QUESTION_PATTERNS[name], body)
        if match:
            plan.update({k: v for k, v in match.groupdict().items() if v is not None})
            plan.update(intent=intent, form=name)
            break
    else:
        if re.search(QUESTION_PATTERNS["description"], body):
            plan.update(intent="description", form="description")
        elif re.search(QUESTION_PATTERNS["summary"], body):
            plan.update(intent="summary", form="summary")
        elif question["type"] in {"unsupported", "relation", "definition"}:
            plan.update(intent="general", form="surface")
    # Date/reason/method expressions are not object wildcards. Their dedicated
    # validators retain responsibility for direction, units and modality.
    if question["type"] not in {"time", "quantity", "reason", "method"}:
        for match in re.finditer(QUESTION_PATTERNS["slot"], body):
            plan["slots"].append({"text": match.group(), "clean_start": start + match.start(),
                                  "clean_end": start + match.end(), "local_start": match.start(),
                                  "local_end": match.end()})
    if context:
        plan["rule_ids"].append("QP102")
    if plan.get("form") in {"material", "event_location", "endpoints"}:
        plan["rule_ids"].append("QP104")
    if plan["intent"] in {"summary", "general", "unsupported"}:
        plan["rule_ids"].append("QP103")
    return plan
