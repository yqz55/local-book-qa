"""Versioned automatic relations plus optional, separately recorded corrections."""

import hashlib
import json

from .tokenization import is_han
from .legacy_aliases import ALIAS_RULES, LEGACY_ALIAS_GROUPS

ALIAS_POLICY = "entity-aliases/1"
LEGACY_ALIAS_QUERY_POLICY = "book-frequency-aliases/2"
CURRENT_QUERY_POLICY = "book-person-rules/3"


def alias_snapshot(groups=None, *, legacy=False):
    groups = (LEGACY_ALIAS_GROUPS if legacy else []) if groups is None else groups
    if not isinstance(groups, list):
        raise ValueError("别名必须是分组的 JSON 数组，每组列出同一人物的不同称呼。")
    normalized, seen = [], set()
    for group in groups:
        if not isinstance(group, list) or any(not isinstance(name, str) for name in group):
            raise ValueError("每个别名组必须是汉字词数组。")
        if any(not 2 <= len(name) <= 32 or not all(is_han(c) for c in name) for name in group):
            raise ValueError("每个别名必须是 2–32 字的纯汉字词。")
        names = sorted(set(group))
        if len(names) < 2:
            raise ValueError("每组至少需要两个不同的名称。")
        if seen.intersection(names):
            raise ValueError("同一名称不能属于多个别名组，请将同一对象的名称放在同一组。")
        seen.update(names)
        normalized.append(names)
    normalized.sort()
    identity = {"policy": ALIAS_POLICY, "groups": normalized, "weight": 1, "expansion_depth": 1}
    sha = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8")).hexdigest()
    return {**identity, "sha256": sha, "rules": ALIAS_RULES if legacy else CURRENT_ALIAS_RULES}


CURRENT_ALIAS_RULES = {
    "A002": "将问题中的完整名称按本书已接受的同一人物关系双向展开；来源、判断规则和原文证据单独保存。每个去重词项权重为 1。",
    "A003": "正文只统计实际出现的不重叠词项；全名内的简称不重复计 TF，规范化构造出的名称不虚构正文出现次数。",
    "A004": "默认无人工名称表。简称对应多个全名或证据不足时不自动合并；人工校正另存并标明来源。",
}


def inferred_alias_snapshot(discovery, manual_groups=None):
    manual = alias_snapshot(manual_groups)["groups"]
    manual_names = {name for group in manual for name in group}
    automatic, suppressed = [], []
    for group in discovery["groups"]:
        if manual_names.intersection(group):
            suppressed.append({"group": group, "reason": "组内名称已有人工校正，整组自动关联暂停，避免传递合并"})
        else:
            automatic.append(group)
    groups = sorted(automatic + manual)
    origins = []
    for number, group in enumerate(groups, 1):
        source = "manual" if group in manual else "automatic"
        origins.append({"group_id": f"G{number:06d}", "group": group, "source": source,
                        "relations": [] if source == "manual" else [edge for edge in discovery["relations"]
                            if edge["a"] in group and edge["b"] in group]})
    identity = {"policy": "corpus-person-aliases/2", "groups": groups,
                "manual_groups": manual, "automatic_groups": automatic,
                "group_sources": origins, "suppressed_automatic_groups": suppressed,
                "discovery_sha256": discovery["sha256"], "weight": 1, "expansion_depth": 1}
    sha = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8")).hexdigest()
    return {**identity, "sha256": sha, "rules": CURRENT_ALIAS_RULES}


def alias_words(snapshot):
    return sorted({name for group in snapshot["groups"] for name in group})


def expand_query(query_counts, snapshot):
    """Keep literal frequencies separate; add each scoring term at most once."""
    sources = {term: [{"kind": "literal", "term": term, "query_frequency": count}]
               for term, count in query_counts.items()}
    lookup = {name: (f"G{i:06d}", group) for i, group in enumerate(snapshot["groups"], 1) for name in group}
    expansions = []
    for original in query_counts:
        if original not in lookup:
            continue
        group_id, group = lookup[original]
        for alias in group:
            if alias == original:
                continue
            added = alias not in sources
            sources.setdefault(alias, []).append({"kind": "alias", "from": original,
                                                 "rule": "A002", "group_id": group_id})
            expansions.append({"source_term": original, "term": alias, "group": group,
                               "group_id": group_id, "rule": "A002", "weight": 1,
                               "added_to_query": added})
            if "group_sources" in snapshot:
                origin = next(item for item in snapshot["group_sources"] if item["group_id"] == group_id)
                expansions[-1].update(origin=origin["source"], relations=origin["relations"])
                sources[alias][-1]["origin"] = origin["source"]
    return sources, expansions
