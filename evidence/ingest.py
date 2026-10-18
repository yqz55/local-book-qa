"""Explicit cleaning, segmentation, and Unicode source coordinates."""

from bisect import bisect_right
from dataclasses import dataclass
import re


RULES = {
    "C001": "仅删除文件开头的 U+FEFF BOM。",
    "C002": "CRLF 或 CR 统一成 LF；每个换行保留对应原文区间。",
    "C003": "行首、行尾的空格、TAB、全角空格删除；段内不压缩空格。",
    "C004": "段内 TAB 和全角空格各替换为一个 ASCII 空格。",
    "S001": "空白行仅作为边界，不生成检索段落。",
    "S002": "每个非空物理行独立分段，不合并相邻行。",
    "S003": "匹配“第…回/章/节/卷/部/篇”的短行标为章节，章节本身也保留。",
    "S004": "超过上限时，在上限内最后一个句末标点之后切分。",
    "S005": "上限内没有句末标点时，按 Unicode 字符数硬切；无重叠。",
}
CHAPTER = re.compile(r"^第[一二三四五六七八九十百千万零〇两0-9]+[回章节卷部篇].*$")
SENTENCE_END = frozenset("。！？；.!?;…")
EDGE_SPACES = " \t\u3000"


@dataclass(frozen=True)
class IngestConfig:
    max_chars: int = 400

    def __post_init__(self):
        if type(self.max_chars) is not int or not 16 <= self.max_chars <= 4000:
            raise ValueError("每段字符上限必须是 16–4000 的整数。")


def decode_txt(data: bytes, encoding: str = "utf-8") -> str:
    if encoding not in {"utf-8", "gb18030"}:
        raise ValueError("请选择明确的编码：utf-8 或 gb18030。")
    if len(data) > 30 * 1024 * 1024:
        raise ValueError("TXT 文件不能超过 30 MiB。")
    try:
        raw = data.decode(encoding, errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"无法按 {encoding} 严格解码（字节 {exc.start}）。请核对文件编码后重新选择。") from exc
    if "\x00" in raw:
        raise ValueError("文本含 NUL 字符，可能是 UTF-16 或非 TXT 文件；请先转为 UTF-8。")
    if not raw.strip("\ufeff \r\n\t\u3000"):
        raise ValueError("TXT 没有可索引的文字。")
    # Do not silently remove unknown controls, negations, numbers, or symbols.
    for i, char in enumerate(raw):
        if ord(char) < 32 and char not in "\t\r\n":
            raise ValueError(f"原文字符 {i + 1} 含不支持的控制字符 U+{ord(char):04X}。")
    return raw


def clean_text(raw: str) -> dict:
    parts, runs, edits = [], [], []
    clean_offset = 0

    def emit(text: str, start: int, end: int):
        nonlocal clean_offset
        if not text:
            return
        length = len(text)
        # Merge only one-to-one runs. CRLF remains its own two-to-one run.
        if (runs and runs[-1][1] == clean_offset and runs[-1][3] == start
                and runs[-1][1] - runs[-1][0] == runs[-1][3] - runs[-1][2]
                and length == end - start):
            runs[-1][1], runs[-1][3] = clean_offset + length, end
        else:
            runs.append([clean_offset, clean_offset + length, start, end])
        parts.append(text)
        clean_offset += length

    def edit(start: int, end: int, replacement: str, rule: str):
        edits.append({"source_start": start, "source_end": end,
                      "before": raw[start:end], "after": replacement, "rule": rule})

    for match in re.finditer(r"[^\r\n]*(?:\r\n|\r|\n|$)", raw):
        if match.start() == match.end():
            continue
        start, end = match.span()
        body_end = end
        while body_end > start and raw[body_end - 1] in "\r\n":
            body_end -= 1
        left = start
        if start == 0 and raw.startswith("\ufeff"):
            edit(0, 1, "", "C001")
            left += 1
        trimmed_left = left
        while left < body_end and raw[left] in EDGE_SPACES:
            left += 1
        if left > trimmed_left:
            edit(trimmed_left, left, "", "C003")
        right = body_end
        while right > left and raw[right - 1] in EDGE_SPACES:
            right -= 1
        if right < body_end:
            edit(right, body_end, "", "C003")
        cursor = left
        for special in re.finditer("[\t\u3000]", raw[left:right]):
            pos = left + special.start()
            emit(raw[cursor:pos], cursor, pos)
            emit(" ", pos, pos + 1)
            edit(pos, pos + 1, " ", "C004")
            cursor = pos + 1
        emit(raw[cursor:right], cursor, right)
        if body_end < end:
            emit("\n", body_end, end)
            if raw[body_end:end] != "\n":
                edit(body_end, end, "\n", "C002")
    return {"text": "".join(parts), "source_runs": runs, "edits": sorted(edits, key=lambda e: e["source_start"])}


class SourceMap:
    """Runs: [clean_start, clean_end, source_start, source_end], half-open."""

    def __init__(self, runs: list[list[int]], raw: str):
        self.runs = runs
        self.starts = [r[0] for r in runs]
        self.line_starts = [0] + [m.end() for m in re.finditer(r"\r\n|\r|\n", raw)]

    def span(self, start: int, end: int) -> tuple[int, int]:
        if end <= start:
            raise ValueError("映射范围必须非空。")
        a = self.runs[bisect_right(self.starts, start) - 1]
        b = self.runs[bisect_right(self.starts, end - 1) - 1]
        source_start = a[2] + (start - a[0] if a[1] - a[0] == a[3] - a[2] else 0)
        source_end = b[2] + (end - b[0] if b[1] - b[0] == b[3] - b[2] else b[3] - b[2])
        return source_start, source_end

    def location(self, start: int, end: int) -> dict:
        a = bisect_right(self.line_starts, start) - 1
        b = bisect_right(self.line_starts, end - 1) - 1
        return {"start_line": a + 1, "start_column": start - self.line_starts[a] + 1,
                "end_line": b + 1, "end_column": end - self.line_starts[b]}


def segment(cleaned: dict, raw: str, config: IngestConfig) -> list[dict]:
    text = cleaned["text"]
    mapping = SourceMap(cleaned["source_runs"], raw)
    paragraphs, chapter = [], "未识别章节"
    for line in re.finditer(r"[^\n]+", text):
        body = line.group()
        if not body.strip():
            continue
        heading = len(body) <= 120 and bool(CHAPTER.fullmatch(body))
        if heading:
            chapter = body
        cursor = line.start()
        while cursor < line.end():
            limit = min(cursor + config.max_chars, line.end())
            if limit == line.end():
                stop, rule = limit, "S003" if heading else "S002"
            else:
                stop = next((i + 1 for i in range(limit - 1, cursor - 1, -1)
                             if text[i] in SENTENCE_END), limit)
                rule = "S004" if text[stop - 1] in SENTENCE_END else "S005"
            start, end = mapping.span(cursor, stop)
            paragraphs.append({"id": f"P{len(paragraphs) + 1:06d}", "text": text[cursor:stop],
                               "clean_start": cursor, "clean_end": stop,
                               "source_start": start, "source_end": end,
                               "location": mapping.location(start, end), "chapter": chapter,
                               "split_rule": rule,
                               "split_reason": RULES[rule], "is_heading": heading})
            cursor = stop
    return paragraphs


def ingest(data: bytes, encoding: str, config: IngestConfig) -> tuple[str, dict, list[dict]]:
    raw = decode_txt(data, encoding)
    cleaned = clean_text(raw)
    return raw, cleaned, segment(cleaned, raw, config)
