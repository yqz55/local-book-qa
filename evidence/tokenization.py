"""Forward maximum dictionary matching with visible, deterministic fallbacks."""

import hashlib
import json
import re
import unicodedata

from .ingest import SourceMap

TOKEN_RULES = {
    "T001": "连续汉字使用词典正向最长匹配；词典只含 2–32 字汉字词。",
    "T002": "未命中词典的汉字按单字回退，不自动补词或扩展同义词。",
    "T003": "ASCII 字母/数字连同内部的 . _ + - 组成词项，字母转小写。",
    "T004": "其他 Unicode 字母、数字、组合标记和符号各自成词。",
    "T005": "空白、Unicode 标点和控制/格式字符不计词长；跳过区间也保留。",
    "Q001": "查询用相同分词器；重复词保留次数供查看，计分时去重，每词权重固定为 1。",
}
ASCII_RUN = re.compile(r"[A-Za-z0-9]+(?:[._+\-][A-Za-z0-9]+)*")
QUERY_RULES = {
    "Q001": "查询清洗与字符回退规则同正文；重复词计数但计分去重，每词权重固定为 1。",
    "Q002": "从左至右，仅比较当前位置匹配的书内多字汉字词；按全书分词词频降序、长度降序、词项字典序选择。单字仅作回退。",
    "Q003": "当前位置无书内多字词时，按已保存的基础与专用词典最长匹配，再按 T002–T005 回退；不跳过未知文字。",
    "Q004": "书内词频来自实际索引的 collection_tf，不把原文中的子串次数当作分词词频；词频只选分词边界，不另加 BM25 权重。",
}
QUERY_POLICY = "book-frequency/1"
PROTECTED_RULES = {
    "A001": "本书显式别名组中的完整名称优先作为整词；同起点按名称长度降序、字典序选择，再对其他文字使用书内词频和固定回退规则。",
}
PERSON_PROTECTION_RULE = "本书规则发现或人工校正的名称优先整词匹配，同起点长名称优先；其他文字按书内词频和固定回退处理。"


def is_han(char: str) -> bool:
    c = ord(char)
    return (0x3400 <= c <= 0x4DBF or 0x4E00 <= c <= 0x9FFF
            or 0xF900 <= c <= 0xFAFF or 0x20000 <= c <= 0x323AF)


class Tokenizer:
    def __init__(self, words: list[str], book_frequencies: dict[str, int] | None = None,
                 protected_words: list[str] | None = None, *, protection_rule: str = "A001"):
        self.protection_rule = protection_rule
        if not isinstance(words, list) or any(not isinstance(w, str) for w in words):
            raise ValueError("词典必须是由汉字词组成的 JSON 数组。")
        if any(not 2 <= len(w) <= 32 or not all(is_han(c) for c in w) for w in words):
            raise ValueError("每个词条必须是 2–32 字的纯汉字词。")
        self.words = sorted(set(words))
        self.sha256 = hashlib.sha256(json.dumps(self.words, ensure_ascii=False,
                                               separators=(",", ":")).encode("utf-8")).hexdigest()
        self.trie = {}
        for word in self.words:
            node = self.trie
            for char in word:
                node = node.setdefault(char, {})
            node[""] = True
        self.book_frequencies = book_frequencies
        self.protected_trie = {}
        for word in protected_words or []:
            node = self.protected_trie
            for char in word:
                node = node.setdefault(char, {})
            node[""] = word
        self.book_trie = {}
        if book_frequencies is not None:
            if (not isinstance(book_frequencies, dict)
                    or any(not isinstance(t, str) or type(f) is not int or f <= 0
                           for t, f in book_frequencies.items())):
                raise ValueError("本书词频必须是词项到正整数的映射。")
            for term in sorted(book_frequencies):
                if 2 <= len(term) <= 32 and all(is_han(c) for c in term):
                    node = self.book_trie
                    for char in term:
                        node = node.setdefault(char, {})
                    node[""] = term

    def tokenize(self, text: str, offset: int = 0, mapping: SourceMap | None = None) -> tuple[list[dict], list[dict]]:
        tokens, skipped = [], []
        i = 0
        while i < len(text):
            char = text[i]
            choices = []
            if is_han(char):
                node, j = self.protected_trie, i
                protected = []
                while j < len(text) and text[j] in node:
                    node = node[text[j]]
                    j += 1
                    if "" in node:
                        word = node[""]
                        protected.append(word)
                if protected:
                    protected.sort(key=lambda word: (-len(word), word))
                    term, rule = protected[0], self.protection_rule
                    end = i + len(term)
                    choices = [{"term": word, "book_tf": (self.book_frequencies or {}).get(word, 0),
                                "length": len(word)} for word in protected]
                else:
                    node, j = self.book_trie, i
                    while j < len(text) and text[j] in node:
                        node = node[text[j]]
                        j += 1
                        if "" in node:
                            word = node[""]
                            choices.append({"term": word, "book_tf": self.book_frequencies[word], "length": len(word)})
                    choices.sort(key=lambda c: (-c["book_tf"], -c["length"], c["term"]))
                    if choices:
                        term, rule = choices[0]["term"], "Q002"
                        end = i + len(term)
                    else:
                        node, j, longest = self.trie, i, i
                        while j < len(text) and text[j] in node:
                            node = node[text[j]]
                            j += 1
                            if "" in node:
                                longest = j
                        end = longest if longest > i else i + 1
                        term, rule = text[i:end], "T001" if longest > i else "T002"
            elif char.isascii() and char.isalnum():
                match = ASCII_RUN.match(text, i)
                end, term, rule = match.end(), match.group().lower(), "T003"
            elif not char.isspace() and unicodedata.category(char)[0] in "LMNS":
                end, term, rule = i + 1, char, "T004"
            else:
                end, term, rule = i + 1, None, "T005"
            item = {"surface": text[i:end], "clean_start": offset + i,
                    "clean_end": offset + end, "rule": rule}
            if mapping is not None:
                item["source_start"], item["source_end"] = mapping.span(offset + i, offset + end)
            if self.book_frequencies is not None and term is not None:
                decision = rule if rule in {"A001", "N001"} else "Q002" if choices else "Q003"
                item.update(book_frequency=self.book_frequencies.get(term, 0),
                            match_candidates=choices, selection_rule=decision,
                            selection_reason={**QUERY_RULES, **PROTECTED_RULES, "N001": PERSON_PROTECTION_RULE}[decision],
                            match_source="person_discovery" if rule == "N001" else "alias_dictionary" if rule == "A001" else "book_vocabulary" if choices else {
                                "T001": "saved_dictionary", "T002": "han_character_fallback",
                                "T003": "ascii_run", "T004": "unicode_fallback"}[rule])
            if term is None:
                # Adjacent ignored characters share one trace interval.
                if skipped and skipped[-1]["clean_end"] == item["clean_start"]:
                    skipped[-1]["surface"] += item["surface"]
                    skipped[-1]["clean_end"] = item["clean_end"]
                    if mapping is not None:
                        skipped[-1]["source_end"] = item["source_end"]
                else:
                    skipped.append(item)
            else:
                tokens.append({"term": term, **item})
            i = end
        return tokens, skipped
