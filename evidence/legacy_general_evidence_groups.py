"""Source-preserving evidence groups; retrieval windows are never factual links."""

from copy import deepcopy
import re

GROUP_POLICY = "source-evidence-groups/2"
GROUP_CONFIG = {"top_groups": 3, "max_context_sentences": 2, "max_group_sentences": 4}
GROUP_RULES = {
    "EG101": "已通过抽取的同一句或相邻两句构成一组；定义、时间、数量、关系、原因或方法主字段所在句为直接证据，目标或对象关联句为关联证据。",
    "EG102": "未形成答案时围绕有内容匹配的句子组织候选；只在同一已召回窗口内补相邻句，不跨已记录边界。",
    "EG103": "补充句需增加查询概念覆盖，或重复明确对象，或含该项目等明确指称；仅作上下文，不能自动变成答案依据。",
    "EG104": "同组每个查询概念只取最高匹配值，覆盖分不累加重叠窗口BM25分；相同原文句集合去重。",
    "EG105": "已被同一证据组覆盖的命中句不再生成重复卡片；不同答案字段的证据仍分别保留。每句单独保存原文区间。",
}


def _coverage(sentences):
    concepts = {}
    for sentence in sentences:
        for feature in sentence["features"]:
            concepts[feature["concept"]] = max(concepts.get(feature["concept"], 0), feature["match_value"])
    total = sum(concepts.values()) / len(concepts) if concepts else 0.0
    return total, [{"concept": key, "match_value": value, "contribution": value / len(concepts)}
                   for key, value in sorted(concepts.items())]


def build_evidence_groups(index, result):
    evidence, answer = result["sentence_evidence"], result["answer"]
    sentences = {s["sentence_id"]: s for s in evidence["candidates"]}
    ordered = [s["id"] for s in index["sentences"]["sentences"]]
    order = {sid: number for number, sid in enumerate(ordered)}
    windows = result["window_retrieval"]["candidates"][:index["answering"]["config"]["window_limit"]]
    blocked = {(b["left_sentence_id"], b["right_sentence_id"]) for b in index["context"]["boundaries"]}
    primary = {"time", "value", "definition", "description", "reason", "method"}
    groups, seen, represented = [], set(), set()

    def append(ids, direct, support, rule, reason, binding=None):
        ids = sorted(set(ids), key=order.__getitem__)
        key = tuple(ids)
        if not ids or key in seen:
            return None
        seen.add(key)
        rows = [sentences[sid] for sid in ids]
        score, details = _coverage(rows)
        origins = [w["window_id"] for w in windows if set(ids).issubset(w["sentence_ids"])]
        spans = []
        for sid in ids:
            sentence = sentences[sid]
            role = "direct" if sid in direct else "support" if sid in support else "context"
            spans.append({"sentence_id": sid, "source_start": sentence["source_start"],
                          "source_end": sentence["source_end"], "clean_start": sentence["clean_start"],
                          "clean_end": sentence["clean_end"], "location": deepcopy(sentence["location"]), "chapter": sentence["chapter"],
                          "role": role, "rule": rule, "reason": reason if role != "context" else "EG103：补充邻近内容，未作为字段关联证明"})
        group = {"id": f"EG{len(groups) + 1:06d}", "sentence_ids": ids,
                 "anchor_sentence_ids": [sid for sid in ids if sentences[sid]["score"] > 0],
                 "direct_sentence_ids": [sid for sid in ids if sid in direct],
                 "supporting_sentence_ids": [sid for sid in ids if sid in support],
                 "context_sentence_ids": [sid for sid in ids if sid not in direct and sid not in support],
                 "source_spans": spans, "window_ids": origins, "coverage": score, "features": details,
                 "binding": deepcopy(binding), "is_answer_support": rule == "EG101", "inclusion_decisions": [{"sentence_id": s["sentence_id"], "role": s["role"],
                     "rule": s["rule"], "reason": s["reason"]} for s in spans],
                 "reason": reason, "source_start": rows[0]["source_start"], "source_end": rows[-1]["source_end"]}
        groups.append(group)
        represented.update(ids)
        return group

    # Accepted candidates survive separately when their values disagree. A
    # duplicate proof interval is shown once regardless of window overlap.
    decisions = [d for d in answer.get("decisions", []) if d.get("accepted")]
    decisions.sort(key=lambda d: (not bool(set(d.get("sentence_ids", [d.get("sentence_id")])).intersection(answer["evidence_ids"])),
                                 min((order.get(sid, 10**9) for sid in d.get("sentence_ids", [d.get("sentence_id")])), default=10**9)))
    for decision in decisions:
        ids = [sid for sid in decision.get("sentence_ids", [decision.get("sentence_id")]) if sid in sentences]
        direct = {field["sentence_id"] for name, field in decision["fields"].items() if name in primary}
        if not direct:
            direct = set(ids)
        append(ids, direct, set(ids) - direct, "EG101", "抽取字段及其对象关联来自所列原句", decision.get("binding"))

    for seed in evidence["candidates"]:
        sid = seed["sentence_id"]
        if sid in represented or seed["score"] <= 0:
            continue
        ids, used = [sid], {f["concept"] for f in seed["features"] if f["matches"]}
        neighbours = []
        for w in windows:
            if sid not in w["sentence_ids"]:
                continue
            for other in w["sentence_ids"]:
                if other != sid and other in sentences and abs(order[other] - order[sid]) == 1:
                    neighbours.append(other)
        neighbours = sorted(set(neighbours), key=lambda other: (abs(order[other] - order[sid]), order[other]))
        for other in neighbours:
            left, right = sorted((sid, other), key=order.__getitem__)
            if (left, right) in blocked or len(ids) >= 1 + GROUP_CONFIG["max_context_sentences"]:
                continue
            neighbour = sentences[other]
            added = {f["concept"] for f in neighbour["features"] if f["matches"]} - used
            target = result["question"].get("target", "")
            repeated = bool(target and target in neighbour["text"] and target in seed["text"])
            reference_match = re.match(r"该(?P<type>项目|试点|活动|实验|工程)", neighbour["text"])
            reference = bool(reference_match and reference_match["type"] in seed["text"])
            if added or repeated or reference:
                ids.append(other)
                used.update(added)
        append(ids, {sid}, set(), "EG102", "有查询内容匹配的原句及按规则加入的邻近上下文")
    for rank, group in enumerate(groups, 1):
        group["rank"] = rank
    answer_groups = [g["id"] for g in groups if answer["evidence_ids"] and g["is_answer_support"] and
                     (set(answer["evidence_ids"]).issubset(g["sentence_ids"]) or
                      (answer["status"] == "ambiguous" and set(answer["evidence_ids"]).intersection(g["sentence_ids"])))]
    answer["evidence_group_ids"] = answer_groups
    top = list(dict.fromkeys([*answer_groups, *(g["id"] for g in groups)]))[:GROUP_CONFIG["top_groups"]]
    return {"policy": GROUP_POLICY, "config": deepcopy(GROUP_CONFIG), "rules": deepcopy(GROUP_RULES),
            "groups": groups, "top_ids": top, "candidate_count": len(groups), "scope": evidence["scope"]}
