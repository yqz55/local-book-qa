"""Offline lexical expansion: explicit senses, book filtering and literal spans.

WordNet senses provide retrieval alternatives, not proof of contextual synonymy.
The auxiliary expression index never changes the main tokenizer, TF or length.
"""

from collections import defaultdict, OrderedDict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

from .tokenization import is_han

ROOT = Path(__file__).resolve().parent.parent
EXPANSION_POLICY = "wordnet-book-expansion/1"
DEFAULT_CONFIG = {"enabled": True, "weight": 0.7, "max_alternatives": 3, "max_sources": 12}
# Grammatical/status words remain literal; they are never removed from queries.
LITERAL_ONLY = frozenset("什么 哪些 怎么 如何 为什么 什么时候 哪里 谁的 是否 一个 这个 那个 这些 那些 我们 你们 他们 她们 自己 已经 正在 可能 应该 可以 不能 不会 不要 不是 没有 并非 未曾 尚未 打算 准备 但是 因为 所以 以及 而且 进行 通过 根据 对于 关于".split())
EXPANSION_RULES = {
    "E001": "只从问题实际文字的完整词项或连续词项短语展开一层；同起点长短语优先，已覆盖词项不重复触发。不在姓名内部匹配，不从扩展词继续展开。",
    "E002": "通用 WordNet 同义集按词义编号分别保留；项目通用校正和本书人工组明确标注来源，人工组暂停相交的通用组。多义匹配是候选，并非完成消歧。",
    "E003": "只接受 2–32 字汉字表达；姓名、语法/否定/状态词及本书停用表达不扩展。候选必须在本书词项边界上连续出现，不跨标点或空白。",
    "E004": "候选按本书人工、项目校正、WordNet 来源优先，再按共享词义数、本书 DF、词长降序及文字升序；每个来源最多保留配置数量，最多处理配置数量的问题表达。",
    "E005": "问题原词及人物别名权重为 1；仅语义扩展的表达使用配置权重。相同表达合并来源且不重复计分；同一查询表达连接的候选组成计分组，每段只保留组内最大贡献，同分按查询词顺序。",
    "E006": "短语 TF/DF 来自独立的真实连续区间；不写入正文词频或段长。完整问题短语计分时，仅用于构成该短语的查询碎片不再单独贡献；问题中独立出现的碎片仍保留。",
}
_COMPILED = OrderedDict()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def valid_expression(value):
    return isinstance(value, str) and 2 <= len(value) <= 32 and all(is_han(c) for c in value)


def normalize_groups(groups):
    if not isinstance(groups, list):
        raise ValueError("表达校正须为分组 JSON 数组。")
    normalized, seen = [], set()
    for group in groups:
        if not isinstance(group, list) or any(not valid_expression(t) or t in LITERAL_ONLY for t in group):
            raise ValueError("每组表达须为 2–32 字汉字词，不能使用语法、否定或状态词。")
        forms = sorted(set(group))
        if len(forms) < 2 or seen.intersection(forms):
            raise ValueError("每组至少两个不同表达，同一表达不能出现在多个校正组。")
        seen.update(forms)
        normalized.append(forms)
    return sorted(normalized)


def check_config(config):
    config = dict(DEFAULT_CONFIG if config is None else config)
    if (config.keys() != DEFAULT_CONFIG.keys() or type(config["enabled"]) is not bool
            or isinstance(config["weight"], bool) or not isinstance(config["weight"], (int, float))
            or not math.isfinite(config["weight"]) or not 0 < config["weight"] <= 1
            or type(config["max_alternatives"]) is not int or not 1 <= config["max_alternatives"] <= 10
            or type(config["max_sources"]) is not int or not 1 <= config["max_sources"] <= 30):
        raise ValueError("扩展配置无效：权重须在 (0,1]，每词候选 1–10，每题表达 1–30，启用项须为布尔值。")
    return config


def resource_snapshot(groups, source, reviewed_groups=None):
    if not isinstance(groups, list) or not isinstance(source, dict):
        raise ValueError("词义资源须包含词义组与来源。")
    checked, ids = [], set()
    for group in groups + ([] if reviewed_groups is None else reviewed_groups):
        if (not isinstance(group, dict) or not isinstance(group.get("id"), str) or group["id"] in ids
                or group.get("pos") not in {"n", "v", "a", "s", "r"}
                or not isinstance(group.get("forms"), list)
                or any(not valid_expression(t) for t in group["forms"])
                or not isinstance(group.get("sense", ""), str)):
            raise ValueError("词义组编号、词性或表达无效。")
        forms = sorted(set(group["forms"]))
        if len(forms) < 2:
            raise ValueError("词义组至少需要两个不同表达。")
        ids.add(group["id"])
        checked.append({"id": group["id"], "pos": group["pos"], "sense": group.get("sense", ""),
                        "forms": forms, "origin": "reviewed" if group in (reviewed_groups or []) else "wordnet"})
    identity = {"policy": "chinese-open-wordnet/1", "source": source,
                "groups": sorted(checked, key=lambda g: g["id"])}
    return {**identity, "sha256": digest(identity)}


@lru_cache(maxsize=1)
def default_resource():
    folder = ROOT / "data" / "wordnet"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    payload = (folder / "wn-data-cmn.tab").read_bytes()
    if hashlib.sha256(payload).hexdigest() != source["files"]["wn-data-cmn.tab"]["sha256"]:
        raise ValueError("随附中文 WordNet 数据校验失败。")
    by_sense = defaultdict(set)
    for line in payload.decode("utf-8").splitlines():
        columns = line.split("\t")
        if (len(columns) == 3 and columns[1] == "cmn:lemma" and valid_expression(columns[2])
                and re.fullmatch(r"\d{8}-[nvasr]", columns[0])):
            by_sense[columns[0]].add(columns[2])
    groups = [{"id": key, "pos": key[-1], "forms": sorted(forms)}
              for key, forms in by_sense.items() if len(forms) >= 2]
    reviewed = json.loads((ROOT / "data" / "query_expansion_overrides.json").read_text(encoding="utf-8"))
    source = {**source, "reviewed_version": reviewed["version"], "reviewed_sha256": digest(reviewed)}
    return resource_snapshot(groups, source, reviewed["groups"])


def _compiled(resource, manual_groups, blocked, names):
    key = (resource["sha256"], tuple(tuple(g) for g in manual_groups), tuple(blocked), tuple(names))
    if key in _COMPILED:
        _COMPILED.move_to_end(key)
        return _COMPILED[key]
    manual_words = {t for g in manual_groups for t in g}
    reviewed = [g for g in resource["groups"] if g["origin"] == "reviewed" and not manual_words.intersection(g["forms"])]
    reviewed_words = {t for g in reviewed for t in g["forms"]}
    groups = [g for g in resource["groups"] if g["origin"] == "wordnet"
              and not (manual_words | reviewed_words).intersection(g["forms"])] + reviewed
    groups += [{"id": f"book:{i:04d}", "pos": "unspecified", "sense": "本书人工校正（词性未自动判断）",
                "forms": forms, "origin": "manual"} for i, forms in enumerate(manual_groups, 1)]
    excluded = set(blocked) | set(names) | LITERAL_ONLY
    lookup = defaultdict(list)
    for group in sorted(groups, key=lambda g: g["id"]):
        for form in group["forms"]:
            if form not in excluded:
                lookup[form].append(group)
    prefixes = {form[:i] for form in lookup for i in range(1, len(form) + 1)}
    result = (dict(lookup), prefixes, excluded)
    _COMPILED[key] = result
    while len(_COMPILED) > 4:
        _COMPILED.popitem(last=False)
    return result


def expression_spans(tokens, lookup, prefixes):
    """Whole-token sequences only; physical gaps and protected names stop a span."""
    for start in range(len(tokens)):
        expression = ""
        for end in range(start, len(tokens)):
            token = tokens[end]
            if (token["rule"] == "N001" or (end > start and tokens[end - 1]["clean_end"] != token["clean_start"])):
                break
            expression += token["surface"]
            if expression not in prefixes:
                break
            if expression in lookup:
                yield expression, start, end + 1


def expansion_snapshot(index, resource=None, config=None, manual_groups=None, blocked=None):
    resource = default_resource() if resource is None else resource
    # Revalidate every supplied snapshot, including its origins, before caching it.
    checked = resource_snapshot([g for g in resource["groups"] if g["origin"] == "wordnet"], resource["source"],
                                [g for g in resource["groups"] if g["origin"] == "reviewed"])
    if checked != resource:
        raise ValueError("词义资源快照校验失败。")
    config, manual_groups = check_config(config), normalize_groups([] if manual_groups is None else manual_groups)
    if blocked is None:
        blocked = []
    if not isinstance(blocked, list) or any(not valid_expression(t) for t in blocked):
        raise ValueError("停用表达须为 2–32 字汉字词数组。")
    blocked = sorted(set(blocked))
    names = sorted(set(index["name_discovery"]["names"]) | {t for g in index["aliases"]["groups"] for t in g})
    lookup, prefixes, excluded = _compiled(resource, manual_groups, blocked, names)
    matches = defaultdict(dict)
    for paragraph in index["paragraphs"]:
        for expression, start, end in expression_spans(paragraph["tokens"], lookup, prefixes):
            if expression not in excluded:
                matches[expression].setdefault(paragraph["id"], []).append([start, end])
    identity = {"policy": EXPANSION_POLICY, "config": config, "resource": resource,
                "manual_groups": manual_groups, "blocked": blocked,
                "book_matches": {term: matches[term] for term in sorted(matches)}, "rules": EXPANSION_RULES}
    return {**identity, "sha256": digest(identity)}


def expand_query(index, tokens, sources):
    """Return scoring features plus every lexical decision; no index mutation."""
    snapshot = index["query_expansion"]
    cfg, resource = snapshot["config"], snapshot["resource"]
    names = sorted(set(index["name_discovery"]["names"]) | {t for g in index["aliases"]["groups"] for t in g})
    lookup, prefixes, excluded = _compiled(resource, snapshot["manual_groups"], snapshot["blocked"], names)
    matches = snapshot["book_matches"]
    decisions, expansions, links, covered = [], [], [], set()
    selected_spans = []
    features = {term: {"term": term, "weight": 1, "kind": "token", "sources": list(origins),
                       "query_frequency": sum(s.get("query_frequency", 0) for s in origins if s["kind"] == "literal")}
                for term, origins in sources.items()}
    if not cfg["enabled"]:
        return list(features.values()), expansions, decisions
    spans = sorted(expression_spans(tokens, lookup, prefixes), key=lambda s: (s[1], -(s[2] - s[1]), -len(s[0]), s[0]))
    used_tokens, source_count = set(), 0
    for expression, start, end in spans:
        if used_tokens.intersection(range(start, end)):
            continue
        used_tokens.update(range(start, end))
        source_count += 1
        groups = lookup[expression]
        candidates = defaultdict(list)
        for group in groups:
            for form in group["forms"]:
                if form != expression:
                    candidates[form].append(group)
        records = []
        for form, shared in candidates.items():
            reason = ("excluded_expression" if form in excluded else "absent_from_book" if form not in matches
                      else "source_limit" if source_count > cfg["max_sources"] else None)
            records.append({"term": form, "sense_ids": sorted(g["id"] for g in shared),
                            "parts_of_speech": sorted({g["pos"] for g in shared}),
                            "origin": min((g["origin"] for g in shared), key={"manual": 0, "reviewed": 1, "wordnet": 2}.get),
                            "book_df": len(matches.get(form, {})), "accepted": False, "reason": reason})
        records.sort(key=lambda r: ({"manual": 0, "reviewed": 1, "wordnet": 2}[r["origin"]],
                                    -len(r["sense_ids"]), -r["book_df"], -len(r["term"]), r["term"]))
        accepted = 0
        span = {"source_term": expression, "query_source_start": tokens[start]["source_start"],
                "query_source_end": tokens[end - 1]["source_end"]}
        selected_spans.append((expression, start, end, span))
        for record in records:
            if record["reason"] is None:
                if accepted >= cfg["max_alternatives"]:
                    record["reason"] = "alternative_limit"
                else:
                    record.update(accepted=True, reason="direct_sense_and_book_match")
                    accepted += 1
                    form = record["term"]
                    added = form not in features
                    origin = {"kind": "lexical", "from": expression, "rule": "E002", "sense_ids": record["sense_ids"],
                              "origin": record["origin"], "query_source_start": span["query_source_start"],
                              "query_source_end": span["query_source_end"]}
                    if added:
                        features[form] = {"term": form, "weight": cfg["weight"], "kind": "expression",
                                          "query_frequency": 0, "sources": []}
                    if origin not in features[form]["sources"]:
                        features[form]["sources"].append(origin)
                    expansions.append({**span, **record, "weight": features[form]["weight"], "added_to_query": added, "rule": "E002"})
                    links.append((expression, form))
        decisions.append({**span, "senses": [{k: g[k] for k in ("id", "pos", "origin", "sense")} for g in groups],
                          "ambiguous": len(groups) > 1, "candidates": records})
        if accepted and expression not in features:
            features[expression] = {"term": expression, "weight": 1, "kind": "expression", "query_frequency": 0, "sources": []}
    # Promote every original phrase after all alternatives are chosen, independent of
    # query order (an earlier phrase can also be added as a later word's alternative).
    literal_counts = defaultdict(int)
    for expression, start, end, span in selected_spans:
        if expression in features and features[expression]["kind"] == "expression":
            literal_counts[expression] += 1
            features[expression]["weight"] = 1
            origin = {"kind": "literal_expression", **span}
            if origin not in features[expression]["sources"]:
                features[expression]["sources"].append(origin)
            covered.update(range(start, end))
    # Literal phrase frequencies and constituent suppression refer only to real query spans.
    for feature in features.values():
        if any(s["kind"] == "literal_expression" for s in feature["sources"]):
            feature["query_frequency"] = literal_counts[feature["term"]]
    for expansion in expansions:
        expansion["weight"] = features[expansion["term"]]["weight"]
    for feature in features.values():
        positions = [i for i, t in enumerate(tokens) if t["term"] == feature["term"]]
        if positions and all(i in covered for i in positions):
            feature["covered_by_expression"] = True
    # This graph only groups scoring features already chosen above; it never generates terms.
    parent = {term: term for term in features}
    def find(term):
        while parent[term] != term:
            parent[term] = parent[parent[term]]
            term = parent[term]
        return term
    for source, target in links:
        parent[find(target)] = find(source)
    grouped = defaultdict(list)
    for term in features:
        grouped[find(term)].append(term)
    number = 0
    for group in grouped.values():
        if len(group) < 2:
            continue
        number += 1
        for term in group:
            features[term]["score_group"] = f"E{number:04d}"
    return list(features.values()), expansions, decisions
