"""Auditable jieba-style dictionary DAG, HMM disabled, frozen book adaptation.

The dictionary is loaded as data from the pinned jieba distribution. No model,
global jieba tokenizer, mutable global word-frequency state, or cache file is used.
"""

from collections import OrderedDict
from functools import lru_cache
import hashlib
from importlib.metadata import distribution, version
import json
import math
import unicodedata

from .tokenization import ASCII_RUN, Tokenizer, is_han

DAG_POLICY = "jieba-dag-book/1"
DAG_RULES = {
    "D001": "使用 jieba 通用词典的全部汉字词和词频构建 DAG；从句尾动态规划，选择整段总路径分数最大的不重叠切分，关闭 HMM。",
    "D002": "当前位置没有词典候选时，仅为该位置加入单字回退边；正常词典中的单字与未登录单字分别记录。",
    "D003": "边分数 = ln(词典频次 / 词典总频次) + 本书奖励；多字词奖励上限由配置固定（默认 0.25），由一次初分词频计算并冻结，正文与查询使用同一配置。",
    "D004": "路径同分时优先选择当前位置较长的词；展示所有候选的基础分、书内奖励、后续最优分与最终路径分。",
    "D005": "用户明确补充的本书专用词条优先完整匹配；与已发现姓名同起点时长词优先，长度相同先用姓名规则。",
    "N001": "已发现姓名或人工关系校正中的名称优先完整匹配；其他文字使用冻结的词典 DAG 与本书奖励规则。",
    "Q001": "正文与查询使用同一冻结分词器；实际重复词保留次数供查看，计分去重，每个词项权重为 1。",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def lexicon_snapshot(frequencies, *, source="controlled", total=None, resource_sha256=None):
    if (not isinstance(frequencies, dict) or any(not isinstance(w, str) or not 1 <= len(w) <= 32
            or not all(is_han(c) for c in w) or type(f) is not int or f <= 0 for w, f in frequencies.items())):
        raise ValueError("通用词典必须包含 1–32 字汉字词及正整数词频。")
    total = max(1, sum(frequencies.values())) if total is None else total
    if type(total) is not int or total < max(1, sum(frequencies.values())):
        raise ValueError("词典总频次不能小于各词频之和。")
    identity = {"source": source, "frequencies": dict(sorted(frequencies.items())), "total": total,
                "resource_sha256": resource_sha256}
    return {**identity, "sha256": _digest(identity)}


@lru_cache(maxsize=1)
def default_lexicon():
    if version("jieba") != "0.42.1":
        raise ValueError("请安装固定版本 jieba==0.42.1 后建立新索引。")
    data = distribution("jieba").locate_file("jieba/dict.txt").read_bytes()
    frequencies, total = {}, 0
    for line in data.decode("utf-8").splitlines():
        word, count, *_ = line.split()
        total += int(count)
        if 1 <= len(word) <= 32 and all(is_han(c) for c in word):
            frequencies[word] = int(count)
    return lexicon_snapshot(frequencies, source="jieba/0.42.1/dict.txt (Han entries)", total=total,
                            resource_sha256=hashlib.sha256(data).hexdigest())


_PREPARED = OrderedDict()


def _prepare(lexicon, words):
    key = (lexicon["sha256"], tuple(words))
    if key in _PREPARED:
        _PREPARED.move_to_end(key)
        return _PREPARED[key]
    frequencies = dict(lexicon["frequencies"])
    for word in words:
        frequencies.setdefault(word, 1)
    total = lexicon["total"] + sum(frequencies[w] for w in words if w not in lexicon["frequencies"])
    prefixes = set()
    for word in frequencies:
        for width in range(1, len(word) + 1):
            prefixes.add(word[:width])
    prepared = frequencies, total, prefixes
    _PREPARED[key] = prepared
    while len(_PREPARED) > 2:
        _PREPARED.popitem(last=False)
    return prepared


class DAGTokenizer:
    def __init__(self, lexicon, words, names=(), book_words=(), seed_frequencies=None,
                 *, book_bonus_cap=0.25, indexed_frequencies=None):
        words = Tokenizer(list(words)).words
        if (not isinstance(book_bonus_cap, (float, int)) or isinstance(book_bonus_cap, bool)
                or not math.isfinite(book_bonus_cap) or not 0 <= book_bonus_cap <= 1):
            raise ValueError("本书路径奖励上限必须为 [0,1] 内的有限数。")
        self.lexicon = lexicon
        self.frequencies, self.total, self.prefixes = _prepare(lexicon, words)
        self.words = sorted(w for w in self.frequencies if len(w) >= 2)
        self.sha256 = _digest({"lexicon_sha256": lexicon["sha256"], "extra_words": words})
        self.names, self.book_words = set(names), set(book_words)
        self.protected = {}
        for word in sorted(self.names | self.book_words):
            node = self.protected
            for char in word:
                node = node.setdefault(char, {})
            node[""] = word
        self.book_bonus_cap = book_bonus_cap
        self.set_book_frequencies(seed_frequencies or {})
        self.indexed_frequencies = indexed_frequencies
        self.last_trace = []

    def set_book_frequencies(self, frequencies):
        self.seed_frequencies = frequencies
        self.book_scale = math.log1p(max(frequencies.values(), default=0)) or 1.0

    def _protected_at(self, text, start):
        node, end, found = self.protected, start, None
        while end < len(text) and text[end] in node:
            node = node[text[end]]
            end += 1
            if "" in node:
                found = node[""]
        return found

    def _plain(self, text, offset, mapping, capture):
        size, dag, route = len(text), {}, {len(text): 0.0}
        logtotal = math.log(self.total)
        for start in range(size):
            ends = []
            for end in range(start + 1, min(size, start + 32) + 1):
                word = text[start:end]
                if word not in self.prefixes:
                    break
                if word in self.frequencies:
                    ends.append(end)
            dag[start] = ends or [start + 1]
        decisions, selected = {}, {}
        for start in range(size - 1, -1, -1):
            choices = []
            for end in dag[start]:
                word = text[start:end]
                freq = self.frequencies.get(word, 0)
                seed_tf = self.seed_frequencies.get(word, 0)
                base = math.log(freq or 1) - logtotal
                bonus = self.book_bonus_cap * math.log1p(seed_tf) / self.book_scale if len(word) >= 2 else 0.0
                path = base + bonus + route[end]
                choices.append({"term": word, "length": len(word), "dictionary_frequency": freq,
                                "seed_tf": seed_tf, "base_score": base, "book_bonus": bonus,
                                "suffix_score": route[end], "path_score": path, "clean_end": offset + end})
            choices.sort(key=lambda c: (-c["path_score"], -c["length"], c["term"]))
            best = choices[0]
            route[start], selected[start], decisions[start] = best["path_score"], best, choices
        if capture:
            nodes = []
            for start in range(size):
                a = offset + start
                source_start = mapping.span(a, a + 1)[0] if mapping else a
                nodes.append({"clean_start": a, "source_start": source_start,
                              "selected_term": selected[start]["term"], "candidates": decisions[start]})
            self.last_trace.append({"clean_start": offset, "clean_end": offset + size,
                                    "path_score": route[0], "nodes": nodes})
        tokens, start = [], 0
        while start < size:
            best = selected[start]
            end, term = start + best["length"], best["term"]
            rule = "D001" if best["dictionary_frequency"] else "D002"
            token = self._item(text[start:end], offset + start, offset + end, rule, mapping)
            token.update(term=term, dictionary_frequency=best["dictionary_frequency"],
                         adaptation_frequency=best["seed_tf"], book_bonus=best["book_bonus"],
                         base_score=best["base_score"], path_score=best["path_score"],
                         book_frequency=(self.indexed_frequencies or self.seed_frequencies).get(term, 0),
                         selection_rule="D004", match_source="general_dictionary" if rule == "D001" else "han_character_fallback")
            if capture:
                token["match_candidates"] = decisions[start]
            tokens.append(token)
            start = end
        return tokens

    @staticmethod
    def _item(surface, start, end, rule, mapping):
        result = {"surface": surface, "clean_start": start, "clean_end": end, "rule": rule}
        if mapping:
            result["source_start"], result["source_end"] = mapping.span(start, end)
        return result

    def tokenize(self, text, offset=0, mapping=None, *, capture=False):
        self.last_trace = []
        tokens, skipped, start = [], [], 0
        while start < len(text):
            char = text[start]
            if is_han(char):
                name = self._protected_at(text, start)
                if name:
                    end = start + len(name)
                    rule = "N001" if name in self.names else "D005"
                    token = self._item(name, offset + start, offset + end, rule, mapping)
                    token.update(term=name, book_frequency=(self.indexed_frequencies or self.seed_frequencies).get(name, 0),
                                 adaptation_frequency=self.seed_frequencies.get(name, 0), selection_rule=rule,
                                 match_source="person_discovery" if rule == "N001" else "book_dictionary")
                    tokens.append(token)
                else:
                    end = start + 1
                    while end < len(text) and is_han(text[end]) and not self._protected_at(text, end):
                        end += 1
                    tokens.extend(self._plain(text[start:end], offset + start, mapping, capture))
            elif char.isascii() and char.isalnum():
                match = ASCII_RUN.match(text, start)
                end = match.end()
                tokens.append({"term": match.group().lower(), **self._item(match.group(), offset + start, offset + end, "T003", mapping)})
            elif not char.isspace() and unicodedata.category(char)[0] in "LMNS":
                end = start + 1
                tokens.append({"term": char, **self._item(char, offset + start, offset + end, "T004", mapping)})
            else:
                end = start + 1
                item = self._item(text[start:end], offset + start, offset + end, "T005", mapping)
                if skipped and skipped[-1]["clean_end"] == item["clean_start"]:
                    skipped[-1]["surface"] += item["surface"]
                    skipped[-1]["clean_end"] = item["clean_end"]
                    if mapping:
                        skipped[-1]["source_end"] = item["source_end"]
                else:
                    skipped.append(item)
            start = end
        return tokens, skipped
