"""Deterministic sentence evidence, independent of retrieval block lengths.

All offsets are half-open Python Unicode character offsets.  A sentence may
overlap several retrieval paragraphs, but never crosses a cleaned physical
line.  ``complete`` is deliberately conservative around quotations.
"""

from bisect import bisect_right
from copy import deepcopy
import re

from .ingest import SourceMap


SENTENCE_POLICY = "physical-line-sentences-v1"
SENTENCE_RULES = {
    "SS001": "从清洗后的每个非空物理行独立切句；章节标题不作为句子证据，不跨行或章节拼接。",
    "SS002": "中文句号、中文或英文问号及叹号作为句界；连续句末标点与紧随其后的闭引号保留在同一句。",
    "SS003": "连续至少两个中文省略号或至少三个英文点作为句界；省略表达标为不完整片段。",
    "SS004": "物理行末没有受支持的句末标点时保留片段，不猜测排版折行。",
    "SS005": "英文句点仅在空白、闭引号或行末前作为句界；数字间句点、常见缩写、单字母缩写和词内句点不切分。",
    "SS006": "引号内标点等待配对闭引号；闭引号后仍有正文时保留后续归属语；嵌套、未配对或含多个内部句界的引语保守标为片段。",
    "SS007": "句子直接取清洗全文区间，忽略检索段落的长度截断；记录所有相交段落与原文精确区间。",
    "SS008": "逗号、分号、冒号不单独作为句界；句间空白不属于证据句。",
}
SENTENCE_CONFIG = {
    "physical_line_only": True,
    "exclude_headings": True,
    "join_retrieval_chunks": True,
    "terminal_punctuation": "。！？!?",
    "chinese_ellipsis_min_chars": 2,
    "ascii_ellipsis_min_chars": 3,
    "ascii_dot_policy": "conservative-abbreviation-aware",
    "quote_policy": "balanced-with-attribution-conservative-fragments",
    "offset_unit": "unicode-code-point",
}

_OPEN_QUOTES = {"“": "”", "‘": "’", "「": "」", "『": "』", '"': '"', "'": "'"}
_CLOSE_QUOTES = frozenset(_OPEN_QUOTES.values())
_TRAILING_BRACKETS = frozenset("）)]】》〉")
_TERMINALS = frozenset(SENTENCE_CONFIG["terminal_punctuation"])
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "no", "vol", "fig", "inc", "ltd", "dept", "approx", "e", "g", "i",
})
SENTENCE_CONFIG.update({
    "ascii_abbreviations": sorted(_ABBREVIATIONS),
    "quote_pairs": dict(_OPEN_QUOTES),
    "apostrophe_policy": "letters-inside-word-and-unpaired-possessive-are-not-quotes",
    "trailing_closing_brackets": "".join(sorted(_TRAILING_BRACKETS)),
})


def _dot_is_terminal(text: str, position: int) -> bool:
    before = text[position - 1] if position else ""
    after = text[position + 1] if position + 1 < len(text) else ""
    if before.isdigit() and after.isdigit():
        return False
    # Adjacent dots are handled by the ellipsis rule, never individually.
    if before == "." or after == ".":
        return False
    if after and not (after.isspace() or after in _CLOSE_QUOTES or after in _TRAILING_BRACKETS):
        return False
    left = position
    while left > 0 and text[left - 1].isascii() and text[left - 1].isalpha():
        left -= 1
    word = text[left:position]
    if word and (len(word) == 1 or word.lower() in _ABBREVIATIONS):
        return False
    return True


def _terminal(text: str, position: int):
    """Return (end, rule, ellipsis) for a supported terminal run."""
    char = text[position]
    if char in _TERMINALS:
        stop = position + 1
        while stop < len(text) and text[stop] in _TERMINALS:
            stop += 1
        return stop, "SS002", False
    if char in {"…", "."}:
        stop = position + 1
        while stop < len(text) and text[stop] == char:
            stop += 1
        minimum = 2 if char == "…" else 3
        if stop - position >= minimum:
            # Include a following sentence terminal in the same run.
            while stop < len(text) and text[stop] in _TERMINALS:
                stop += 1
            return stop, "SS003", True
        if char == "." and _dot_is_terminal(text, position):
            return stop, "SS005", False
    return None


def _line_sentences(text: str):
    """Yield (start, end, terminal rule, fragment reasons) within one line."""
    start, position = 0, 0
    stack, reasons = [], set()
    quoted_terminals = 0
    while position < len(text):
        char = text[position]
        before = text[position - 1] if position else ""
        after = text[position + 1] if position + 1 < len(text) else ""
        if (char in {"'", "’"} and before.isalnum()
                and (after.isalnum() or not stack or stack[-1] != char)):
            position += 1  # don't / don’t / students' are not quotation boundaries.
            continue
        if char in _OPEN_QUOTES:
            if stack and stack[-1] == char:
                stack.pop()  # The second straight quote closes the first.
            else:
                if stack:
                    reasons.add("nested_quotation")
                stack.append(_OPEN_QUOTES[char])
            position += 1
            continue
        if char in _CLOSE_QUOTES:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                reasons.add("unmatched_closing_quote")
            position += 1
            continue
        terminal = _terminal(text, position)
        if terminal is None:
            position += 1
            continue
        stop, rule, ellipsis = terminal
        inside_quote = bool(stack)
        if ellipsis:
            reasons.add("ellipsis")
        if inside_quote:
            quoted_terminals += 1
            if quoted_terminals > 1:
                reasons.add("multiple_quoted_sentences")
        # A terminal and its closing quotes/brackets form a single span.
        while stop < len(text):
            trailing = text[stop]
            if stack and trailing == stack[-1]:
                stack.pop()
                stop += 1
            elif trailing in _CLOSE_QUOTES and trailing not in {'"', "'"}:
                reasons.add("unmatched_closing_quote")
                stop += 1
            elif trailing in _TRAILING_BRACKETS:
                stop += 1
            else:
                break
        if stack:
            position = stop
            continue
        if inside_quote and text[stop:].strip():
            # “去哪儿？”小明问。must retain the question and speaker together.
            # This can also be independent narration, so extraction must abstain.
            reasons.add("quoted_ending_followed_by_text")
            position = stop
            continue
        left, right = start, stop
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if right > left:
            yield left, right, rule, sorted(reasons)
        start, position = stop, stop
        stack, reasons, quoted_terminals = [], set(), 0
    if start < len(text):
        left, right = start, len(text)
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if right > left:
            reasons.add("no_terminal_punctuation")
            if stack:
                reasons.add("unclosed_quotation")
            yield left, right, "SS004", sorted(reasons)


def build_sentences(index: dict) -> dict:
    """Build an auditable frozen sentence snapshot using an existing index.

    Only stored cleaning, original source, and paragraph spans are read.  No
    lexicon, NLP model, external service, or mutable resource is consulted.
    """
    cleaned, raw = index["cleaning"], index["source"]["text"]
    text = cleaned["text"]
    mapping = SourceMap(cleaned["source_runs"], raw)
    paragraphs = sorted(index["paragraphs"], key=lambda p: (p["clean_start"], p["clean_end"], p["id"]))
    ends = [p["clean_end"] for p in paragraphs]

    def overlaps(start, end):
        cursor = bisect_right(ends, start)
        matches = []
        while cursor < len(paragraphs) and paragraphs[cursor]["clean_start"] < end:
            matches.append(paragraphs[cursor])
            cursor += 1
        return matches

    sentences = []
    for line in re.finditer(r"[^\n]+", text):
        line_paragraphs = overlaps(line.start(), line.end())
        if not line.group().strip() or any(p.get("is_heading", False) for p in line_paragraphs):
            continue
        for local_start, local_end, rule, reasons in _line_sentences(line.group()):
            start, end = line.start() + local_start, line.start() + local_end
            source_start, source_end = mapping.span(start, end)
            matching = overlaps(start, end)
            complete = not reasons
            sentences.append({
                "id": f"S{len(sentences) + 1:06d}",
                "text": text[start:end],
                "clean_start": start, "clean_end": end,
                "source_start": source_start, "source_end": source_end,
                "source_text": raw[source_start:source_end],
                "location": mapping.location(source_start, source_end),
                "chapter": matching[0].get("chapter", "未识别章节") if matching else "未识别章节",
                "paragraph_ids": [p["id"] for p in matching if not p.get("is_heading", False)],
                "boundary_rule": rule,
                "complete": complete, "is_fragment": not complete,
                "fragment_reasons": reasons,
            })
    return {"policy": SENTENCE_POLICY, "rules": deepcopy(SENTENCE_RULES),
            "config": deepcopy(SENTENCE_CONFIG), "sentences": sentences}
