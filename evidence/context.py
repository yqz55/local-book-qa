"""Independent, source-preserving sentence windows and auditable BM25.

The paragraph index remains unchanged.  Windows reuse its frozen token
occurrences; their statistics belong to a separate corpus.  A window's text
is always a real contiguous source range, including any original newlines.
"""

from bisect import bisect_left, bisect_right
from collections import Counter
from copy import deepcopy
import math
import re

from .ingest import SourceMap
from .legacy_retrieval import FORMULA


CONTEXT_POLICY = "source-sentence-windows/1"
CONTEXT_CONFIG = {"min_window_sentences": 2, "max_sentences": 4, "max_chars": 600,
                  "subheading_max_chars": 60}
_TOKEN_FIELDS = ("term", "surface", "clean_start", "clean_end", "source_start", "source_end")
CONTEXT_RULES = {
    "CW001": "保留每个原句，按配置生成连续多句的独立检索窗口；默认 2–4 句且最多 600 个清洗后字符（含原换行）。长单句保留全文，不强切或与邻句组合。",
    "CW002": "普通换行或空白行不阻止完整句组成窗口；展示实际清洗文本和原文连续区间，逐句保存位置，不拼造句子。",
    "CW003": "不跨章节、章节标题或未被句子覆盖的非空文字；短独立无句末标点行按保守小标题处理，阻止两侧组合。",
    "CW004": "编号、项目符号或 Markdown 标题行保留独立出处；不跨此行边缘组合，同行的同一列表项内可保留多句窗口。",
    "CW005": "不完整片段或带引语的句子仅保留单句单位，不与邻句组合；不解析代词、补齐片段或推定跨引语关系。",
    "CW006": "窗口只复用原段落的完整词项区间，同一实际词项出现每窗口只计一次；重叠窗口各自属于独立文档，重新统计 TF、DF、N 与平均段长。",
    "CW007": "短语只继承原表达索引的真实连续词项区间，不跨空白、标点、句子或检索块缺口重建表达；短语不增加词项段长。",
    "CW008": "窗口使用冻结 BM25 参数及问题提炼后的查询特征；同表达组只取最大贡献，已被完整表达覆盖的碎片不计分；按分数、原文起点、窗口编号排序。",
}

_LIST_START = re.compile(
    r"^(?:[-*+•●▪◦]\s+|\d{1,3}[.)、．](?!\d)\s*|"
    r"[（(][一二三四五六七八九十百0-9]+[)）]\s*|"
    r"[一二三四五六七八九十百]+、\s*)"
)
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+\S")


def _config(config):
    checked = dict(CONTEXT_CONFIG if config is None else config)
    if (checked.keys() != CONTEXT_CONFIG.keys()
            or any(type(value) is not int for value in checked.values())
            or checked["min_window_sentences"] != 2
            or not 2 <= checked["max_sentences"] <= 8
            or not 16 <= checked["max_chars"] <= 4000
            or not 1 <= checked["subheading_max_chars"] <= 120):
        raise ValueError("句子窗口配置无效：最少 2 句，最多 2–8 句，字符上限 16–4000，小标题上限 1–120。")
    return checked


def _quoted(text):
    for position, char in enumerate(text):
        if char in "“”‘’「」『』\"'":
            # In-word apostrophes are not quotations (don't / don’t).
            if (char in "'’" and position and position + 1 < len(text)
                    and text[position - 1].isalnum() and text[position + 1].isalnum()):
                continue
            return True
    return False


def _span(sentence):
    return {"sentence_id": sentence["id"], **{key: deepcopy(sentence[key]) for key in (
        "clean_start", "clean_end", "source_start", "source_end", "text", "source_text", "location")}}


def build_context_index(index, config=None):
    """Build a deterministic window snapshot without modifying its input."""
    cfg = _config(config)
    cleaned, raw = index["cleaning"], index["source"]["text"]
    text = cleaned["text"]
    mapping = SourceMap(cleaned["source_runs"], raw)
    sentences = sorted(index["sentences"]["sentences"], key=lambda s: (s["clean_start"], s["id"]))
    starts = [s["clean_start"] for s in sentences]
    paragraphs = {p["id"]: p for p in index["paragraphs"]}
    lines = list(re.finditer(r"[^\n]+", text))
    line_starts = [line.start() for line in lines]
    line_info = []
    for line in lines:
        body = line.group().strip()
        line_info.append({
            "start": line.start(), "end": line.end(),
            "list_or_markdown": bool(_LIST_START.match(body) or _MARKDOWN_HEADING.match(body)),
        })
    metadata = []
    sentence_tokens, sentence_expressions = [], [{} for _ in sentences]
    all_tokens = sorted(
        (token for paragraph in index["paragraphs"] for token in paragraph.get("tokens", [])),
        key=lambda token: (token["clean_start"], token["clean_end"], token["term"]),
    )
    token_starts = [token["clean_start"] for token in all_tokens]
    for sentence in sentences:
        line_position = bisect_right(line_starts, sentence["clean_start"]) - 1
        info = line_info[line_position] if line_position >= 0 else {}
        standalone = bool(info and text[info["start"]:info["end"]].strip() == sentence["text"])
        metadata.append({
            "line_position": line_position,
            "list_or_markdown": bool(info.get("list_or_markdown")),
            "subheading": standalone and len(sentence["text"]) <= cfg["subheading_max_chars"]
                          and sentence.get("is_fragment", not sentence.get("complete", False)),
            "fragment": not sentence.get("complete", False),
            "quotation": _quoted(sentence["text"]),
        })
        cursor = bisect_left(token_starts, sentence["clean_start"])
        tokens = []
        while cursor < len(all_tokens) and all_tokens[cursor]["clean_start"] < sentence["clean_end"]:
            token = all_tokens[cursor]
            if token["clean_end"] <= sentence["clean_end"]:
                tokens.append(token)
            cursor += 1
        sentence_tokens.append(tokens)

    # Preserve only actual original expression matches, never manufacture a
    # phrase by concatenating window tokens over a physical or punctuation gap.
    for term, matches in sorted(index.get("query_expansion", {}).get("book_matches", {}).items()):
        for paragraph_id, positions in sorted(matches.items()):
            tokens = paragraphs[paragraph_id].get("tokens", [])
            for first, stop in positions:
                if not 0 <= first < stop <= len(tokens):
                    continue
                chosen = tokens[first:stop]
                start, end = chosen[0]["clean_start"], chosen[-1]["clean_end"]
                sentence_position = bisect_right(starts, start) - 1
                if (sentence_position < 0 or end > sentences[sentence_position]["clean_end"]
                        or text[start:end] != term
                        or any(left["clean_end"] != right["clean_start"] for left, right in zip(chosen, chosen[1:]))):
                    continue
                source_start, source_end = mapping.span(start, end)
                occurrence = {"clean_start": start, "clean_end": end,
                              "source_start": source_start, "source_end": source_end,
                              "original_paragraph_id": paragraph_id, "token_start": first, "token_end": stop}
                occurrences = sentence_expressions[sentence_position].setdefault(term, [])
                if occurrence not in occurrences:
                    occurrences.append(occurrence)

    boundaries, blocked = [], set()
    for position, (left, right) in enumerate(zip(sentences, sentences[1:])):
        reasons = []
        left_meta, right_meta = metadata[position:position + 2]
        gap = text[left["clean_end"]:right["clean_start"]]
        if left.get("chapter") != right.get("chapter") or gap.strip() or left_meta["subheading"] or right_meta["subheading"]:
            reasons.append("CW003")
        if (left_meta["line_position"] != right_meta["line_position"]
                and (left_meta["list_or_markdown"] or right_meta["list_or_markdown"])):
            reasons.append("CW004")
        if left_meta["fragment"] or right_meta["fragment"] or left_meta["quotation"] or right_meta["quotation"]:
            reasons.append("CW005")
        if reasons:
            blocked.add(position)
            # Adjacent complete sentences can touch; map their encompassing
            # range instead of asking SourceMap to map an empty gap.
            a, b = left["clean_end"], right["clean_start"]
            source_start = left["source_end"]
            source_end = right["source_start"]
            boundaries.append({"left_sentence_id": left["id"], "right_sentence_id": right["id"],
                               "clean_start": a, "clean_end": b, "source_start": source_start,
                               "source_end": source_end, "reasons": reasons})

    units, postings, expression_matches = [], {}, {}
    collection_tf, document_frequency, lengths = Counter(), Counter(), []
    for start_position, first in enumerate(sentences):
        for count in range(1, cfg["max_sentences"] + 1):
            last_position = start_position + count - 1
            if last_position >= len(sentences):
                break
            last = sentences[last_position]
            if count > 1 and (last_position - 1 in blocked
                              or last["clean_end"] - first["clean_start"] > cfg["max_chars"]):
                break
            if count != 1 and count < cfg["min_window_sentences"]:
                continue
            selected = sentences[start_position:last_position + 1]
            unique_tokens = {}
            for original_tokens in sentence_tokens[start_position:last_position + 1]:
                for token in original_tokens:
                    unique_tokens.setdefault((token["clean_start"], token["clean_end"], token["term"]), token)
            # Dictionary/DAG explanations stay once in the original paragraph
            # index.  Overlapping windows only need terms and source ranges.
            tokens = [{key: token[key] for key in _TOKEN_FIELDS} for _, token in sorted(unique_tokens.items())]
            frequencies = dict(sorted(Counter(token["term"] for token in tokens).items()))
            unit_id = f"W{len(units) + 1:06d}"
            source_start, source_end = mapping.span(first["clean_start"], last["clean_end"])
            paragraph_ids = list(dict.fromkeys(pid for sentence in selected for pid in sentence["paragraph_ids"]))
            unit = {"id": unit_id, "unit_id": unit_id, "window_id": unit_id,
                    "kind": "sentence" if count == 1 else "window", "sentence_count": count,
                    "sentence_ids": [sentence["id"] for sentence in selected], "paragraph_ids": paragraph_ids,
                    "sentence_spans": [_span(sentence) for sentence in selected],
                    "text": text[first["clean_start"]:last["clean_end"]],
                    "source_text": raw[source_start:source_end],
                    "clean_start": first["clean_start"], "clean_end": last["clean_end"],
                    "source_start": source_start, "source_end": source_end,
                    "location": mapping.location(source_start, source_end), "chapter": first.get("chapter", "未识别章节"),
                    "tokens": tokens, "term_frequencies": frequencies, "length": len(tokens), "indexed": bool(tokens)}
            units.append(unit)
            if tokens:
                lengths.append(len(tokens))
            for term, frequency in frequencies.items():
                postings.setdefault(term, []).append(unit_id)
                collection_tf[term] += frequency
                document_frequency[term] += 1
            for expressions in sentence_expressions[start_position:last_position + 1]:
                for term, occurrences in expressions.items():
                    target = expression_matches.setdefault(term, {}).setdefault(unit_id, [])
                    for occurrence in occurrences:
                        if occurrence not in target:
                            target.append(deepcopy(occurrence))

    return {"policy": CONTEXT_POLICY, "config": cfg, "rules": deepcopy(CONTEXT_RULES),
            "units": units, "postings": dict(sorted(postings.items())),
            "expression_matches": dict(sorted(expression_matches.items())), "boundaries": boundaries,
            "statistics": {"N": len(lengths), "unit_count": len(units), "sentence_count": len(sentences),
                           "singleton_count": sum(unit["sentence_count"] == 1 for unit in units),
                           "window_count": sum(unit["sentence_count"] > 1 for unit in units),
                           "excluded_empty_token_units": len(units) - len(lengths), "total_tokens": sum(lengths),
                           "avgdl": sum(lengths) / len(lengths) if lengths else 0.0,
                           "vocabulary_size": len(document_frequency),
                           "collection_tf": dict(sorted(collection_tf.items())),
                           "document_frequency": dict(sorted(document_frequency.items())),
                           "expression_document_frequency": {term: len(matches) for term, matches in sorted(expression_matches.items())}}}


def boundary_between(index, left, right):
    """Return recorded rule IDs blocking two adjacent sentences, if any."""
    snapshot = index.get("context") or build_context_index(index)
    left_id = left["id"] if isinstance(left, dict) else left
    right_id = right["id"] if isinstance(right, dict) else right
    return next((list(boundary["reasons"]) for boundary in snapshot["boundaries"]
                 if boundary["left_sentence_id"] == left_id and boundary["right_sentence_id"] == right_id), [])


def search_windows(index, result):
    """Score independent windows using the existing query plan and parameters."""
    snapshot = index.get("context") or build_context_index(index)
    stats, config = snapshot["statistics"], dict(result["config"])
    N, avgdl = stats["N"], stats["avgdl"]
    terms, ids = [], set()
    for feature in result["query_terms"]:
        term = feature["term"]
        expression = feature.get("kind", "token") == "expression"
        posting = snapshot["expression_matches"].get(term, {}) if expression else snapshot["postings"].get(term, [])
        df = len(posting)
        terms.append({**deepcopy(feature), "df": df, "idf": math.log1p((N - df + 0.5) / (df + 0.5)),
                      "in_vocabulary": df > 0})
        if not feature.get("covered_by_expression"):
            ids.update(posting)
    candidates = []
    for unit in snapshot["units"]:
        if unit["id"] not in ids or not unit["indexed"]:
            continue
        norm = 1 - config["b"] + config["b"] * unit["length"] / avgdl
        details = []
        for term in terms:
            if term.get("kind", "token") == "expression":
                occurrences = deepcopy(snapshot["expression_matches"].get(term["term"], {}).get(unit["id"], []))
            else:
                occurrences = [{key: token[key] for key in ("clean_start", "clean_end", "source_start", "source_end")}
                               for token in unit["tokens"] if token["term"] == term["term"]]
            tf = len(occurrences)
            numerator, denominator = tf * (config["k1"] + 1), tf + config["k1"] * norm
            contribution = term["weight"] * term["idf"] * numerator / denominator if tf else 0.0
            details.append({**term, "tf": tf, "matched": tf > 0, "numerator": numerator,
                            "denominator": denominator, "raw_contribution": contribution,
                            "contribution": contribution, "occurrences": occurrences})
        winners = {}
        for position, detail in enumerate(details):
            if detail.get("covered_by_expression"):
                detail.update(contribution=0.0, score_reason="covered_by_expression")
            elif "score_group" in detail:
                group = detail["score_group"]
                if group not in winners or detail["raw_contribution"] > details[winners[group]]["raw_contribution"]:
                    winners[group] = position
        for position, detail in enumerate(details):
            if "score_group" in detail and not detail.get("covered_by_expression"):
                selected = winners[detail["score_group"]] == position
                detail.update(contribution=detail["raw_contribution"] if selected else 0.0,
                              score_reason="group_maximum" if selected else "same_group_not_selected")
        if not any(detail["contribution"] > 0 for detail in details):
            continue
        metadata = {key: deepcopy(value) for key, value in unit.items()
                    if key not in {"tokens", "term_frequencies", "length", "indexed"}}
        candidates.append({**metadata, "dl": unit["length"], "length_normalization": norm,
                           "term_details": details, "score": sum(detail["contribution"] for detail in details)})
    candidates.sort(key=lambda candidate: (-candidate["score"], candidate["source_start"], candidate["window_id"]))
    for rank, candidate in enumerate(candidates, 1):
        candidate["rank"] = rank
    return {"config": config, "window_config": deepcopy(snapshot["config"]), "policy": snapshot["policy"],
            "rules": deepcopy(snapshot["rules"]), "query_terms": terms, "candidates": candidates,
            "top_ids": [candidate["window_id"] for candidate in candidates[:config["top_k"]]],
            "candidate_count": len(candidates),
            "corpus": {"N": N, "avgdl": avgdl, "index_id": index.get("index_id"), "unit": "sentence_window"},
            "formula": {**FORMULA, "score": "sum(actual contributions); each score_group keeps its maximum weighted BM25 contribution; covered query fragments contribute zero",
                        "tie_break": "score descending, source_start ascending, window_id ascending"}}
