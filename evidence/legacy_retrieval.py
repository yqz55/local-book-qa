"""The score used for sorting is exactly the sum of the recorded contributions."""

import base64
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import unicodedata

from . import ALIAS_ENGINE_VERSION, PERSON_ENGINE_VERSION as ENGINE_VERSION, LEGACY_ENGINE_VERSION, SUPPORTED_ENGINE_VERSIONS, schema_version
from .aliases import (ALIAS_RULES, CURRENT_ALIAS_RULES, CURRENT_QUERY_POLICY, LEGACY_ALIAS_GROUPS,
                      LEGACY_ALIAS_QUERY_POLICY, alias_snapshot, alias_words, expand_query, inferred_alias_snapshot)
from .ingest import IngestConfig, RULES, SourceMap, ingest
from .person_names import PERSON_RULES, discover_people
from .tokenization import QUERY_POLICY, QUERY_RULES, TOKEN_RULES, Tokenizer

FORMULA = {
    "idf": "ln(1 + (N - df + 0.5) / (df + 0.5))",
    "length_normalization": "1 - b + b * dl / avgdl",
    "contribution": "weight * idf * tf * (k1 + 1) / (tf + k1 * length_normalization)",
    "score": "sum(contribution for each unique query term, in first-occurrence order)",
    "tie_break": "score descending, source_start ascending, paragraph_id ascending",
}


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 5

    def __post_init__(self):
        if isinstance(self.k1, bool) or not isinstance(self.k1, (int, float)) or not math.isfinite(self.k1) or not 0 < self.k1 <= 10:
            raise ValueError("k1 必须在 (0, 10] 内，且为有限数。")
        if isinstance(self.b, bool) or not isinstance(self.b, (int, float)) or not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("b 必须在 [0, 1] 内，且为有限数。")
        if type(self.top_k) is not int or self.top_k != 5:
            raise ValueError("本版本固定返回最多 5 个相关段落。")


def build_index(data: bytes, filename: str, encoding: str, words: list[str],
                config: IngestConfig | None = None, *, book_words: list[str] | None = None,
                alias_groups: list[list[str]] | None = None, person_config: dict | None = None,
                _engine_version: str = ENGINE_VERSION) -> dict:
    if _engine_version not in SUPPORTED_ENGINE_VERSIONS:
        raise ValueError("不支持的索引引擎版本。")
    legacy = _engine_version == LEGACY_ENGINE_VERSION
    current = _engine_version == ENGINE_VERSION
    alias_legacy = _engine_version == ALIAS_ENGINE_VERSION
    if legacy and book_words:
        raise ValueError("旧版索引不支持单独的书籍专用词条。")
    config = config or IngestConfig()
    raw, cleaned, paragraphs = ingest(data, encoding, config)
    base = Tokenizer(words)
    additions = Tokenizer(book_words if book_words is not None else [])
    discovery = discover_people(cleaned, raw, person_config) if current else None
    aliases = inferred_alias_snapshot(discovery, alias_groups) if current else alias_snapshot(alias_groups, legacy=True) if alias_legacy else None
    names = sorted(set(alias_words(aliases)) | set(discovery["names"] if discovery else [])) if aliases else []
    tokenizer = Tokenizer(sorted(set(base.words) | set(additions.words) | set(names)), protected_words=names,
                          protection_rule="N001" if current else "A001")
    mapping = SourceMap(cleaned["source_runs"], raw)
    postings, collection_tf, document_frequency = {}, Counter(), Counter()
    lengths = []
    for p in paragraphs:
        tokens, skipped = tokenizer.tokenize(p["text"], p["clean_start"], mapping)
        frequencies = dict(sorted(Counter(t["term"] for t in tokens).items()))
        p.update(tokens=tokens, skipped=skipped, term_frequencies=frequencies,
                 length=len(tokens), indexed=bool(tokens))
        if tokens:
            lengths.append(len(tokens))
        for term, count in frequencies.items():
            postings.setdefault(term, []).append(p["id"])
            collection_tf[term] += count
            document_frequency[term] += 1
    if not lengths:
        raise ValueError("文本没有可检索词项（可能只含标点或空白）。")
    identity = {"engine_version": _engine_version, "bytes_sha256": hashlib.sha256(data).hexdigest(),
                "encoding": encoding, "ingest_config": asdict(config),
                "dictionary_sha256": tokenizer.sha256, "unicode_version": unicodedata.unidata_version}
    dictionary = {"words": tokenizer.words, "sha256": tokenizer.sha256}
    if not legacy:
        dictionary.update(base_words=base.words, book_words=additions.words)
        identity.update(base_dictionary_sha256=base.sha256, book_dictionary_sha256=additions.sha256)
    if current or alias_legacy:
        identity["alias_sha256"] = aliases["sha256"]
    if current:
        identity["person_discovery_sha256"] = discovery["sha256"]
    result = {"schema": f"evidence-index/{schema_version(_engine_version)}", "index_id": digest(identity), "identity": identity,
            "source": {"filename": filename, "encoding": encoding, "byte_length": len(data),
                       "bytes_base64": base64.b64encode(data).decode("ascii"),
                       "text": raw, "text_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()},
            "dictionary": dictionary,
            "rules": {**RULES, **TOKEN_RULES, **({} if legacy else QUERY_RULES),
                      **(ALIAS_RULES if alias_legacy else CURRENT_ALIAS_RULES if current else {}),
                      **(PERSON_RULES if current else {})},
            "cleaning": cleaned, "paragraphs": paragraphs,
            "statistics": {"N": len(lengths), "paragraph_count": len(paragraphs),
                           "excluded_empty_token_paragraphs": len(paragraphs) - len(lengths),
                           "total_tokens": sum(lengths), "avgdl": sum(lengths) / len(lengths),
                           "vocabulary_size": len(document_frequency),
                           "collection_tf": dict(sorted(collection_tf.items())),
                           "document_frequency": dict(sorted(document_frequency.items()))},
            "postings": dict(sorted(postings.items()))}
    if current or alias_legacy:
        result["aliases"] = aliases
    if current:
        result["name_discovery"] = discovery
    return result


def _manual_groups(index):
    aliases = index.get("aliases", {})
    if "manual_groups" in aliases:
        return aliases["manual_groups"]
    groups = aliases.get("groups", [])
    # Remove the old shipped example; preserve any other explicit correction.
    default_group = sorted(LEGACY_ALIAS_GROUPS[0])
    return [group for group in groups if group != default_group]


def rebuild_with_book_words(index: dict, book_words: list[str]) -> dict:
    """Rebuild both document boundaries and book TF; keep the source and base lexicon."""
    source, dictionary = index["source"], index["dictionary"]
    return build_index(base64.b64decode(source["bytes_base64"], validate=True), source["filename"],
                       source["encoding"], dictionary.get("base_words", dictionary["words"]),
                       IngestConfig(**index["identity"]["ingest_config"]), book_words=book_words,
                       alias_groups=_manual_groups(index),
                       person_config=index.get("name_discovery", {}).get("config"))


def rebuild_with_aliases(index: dict, groups: list[list[str]] | None) -> dict:
    source, dictionary = index["source"], index["dictionary"]
    return build_index(base64.b64decode(source["bytes_base64"], validate=True), source["filename"],
                       source["encoding"], dictionary.get("base_words", dictionary["words"]),
                       IngestConfig(**index["identity"]["ingest_config"]),
                       book_words=dictionary.get("book_words", []), alias_groups=groups,
                       person_config=index.get("name_discovery", {}).get("config"))


def prepare_index_for_search(index: dict) -> dict:
    if index["identity"]["engine_version"] == ENGINE_VERSION:
        return index
    if index["identity"]["engine_version"] not in SUPPORTED_ENGINE_VERSIONS:
        raise ValueError("不支持该索引的引擎版本。")
    return rebuild_with_aliases(index, _manual_groups(index))


def search(index: dict, query: str, config: BM25Config | None = None, *, _legacy: bool = False,
           _engine_version: str | None = None) -> dict:
    config = config or BM25Config()
    if not isinstance(query, str) or len(query) > 500:
        raise ValueError("问题最多 500 个 Unicode 字符。")
    engine = LEGACY_ENGINE_VERSION if _legacy else _engine_version or ENGINE_VERSION
    if engine not in SUPPORTED_ENGINE_VERSIONS:
        raise ValueError("不支持该查询引擎版本。")
    legacy = engine == LEGACY_ENGINE_VERSION
    current = engine == ENGINE_VERSION
    alias_legacy = engine == ALIAS_ENGINE_VERSION
    has_aliases = current or alias_legacy
    if current:
        index = prepare_index_for_search(index)
    stats = index["statistics"]
    protected = sorted(set(alias_words(index["aliases"])) | set(index["name_discovery"]["names"])) if current else alias_words(index["aliases"]) if alias_legacy else None
    tokenizer = Tokenizer(index["dictionary"]["words"], None if legacy else stats["collection_tf"],
                          protected, protection_rule="N001" if current else "A001")
    # Apply exactly the document cleaning rules to the query, including BOM/CRLF.
    from .ingest import clean_text
    query_cleaning = clean_text(query)
    query_mapping = SourceMap(query_cleaning["source_runs"], query)
    tokens, skipped = tokenizer.tokenize(query_cleaning["text"], mapping=query_mapping)
    query_counts = Counter(t["term"] for t in tokens)
    term_sources, expansions = expand_query(query_counts, index["aliases"]) if has_aliases else ({term: [] for term in query_counts}, [])
    N, avgdl = stats["N"], stats["avgdl"]
    query_terms, candidate_ids = [], set()
    for term in term_sources:
        count = query_counts.get(term, 0)
        df = stats["document_frequency"].get(term, 0)
        idf = math.log1p((N - df + 0.5) / (df + 0.5))
        query_terms.append({"term": term, "query_frequency": count, "weight": 1,
                            "df": df, "idf": idf, "in_vocabulary": df > 0})
        if has_aliases:
            query_terms[-1]["sources"] = term_sources[term]
        candidate_ids.update(index["postings"].get(term, []))
    candidates = []
    for p in index["paragraphs"]:
        if p["id"] not in candidate_ids:
            continue
        norm = 1 - config.b + config.b * p["length"] / avgdl
        details = []
        for term in query_terms:
            tf = p["term_frequencies"].get(term["term"], 0)
            numerator = tf * (config.k1 + 1)
            denominator = tf + config.k1 * norm
            # Missing terms contribute zero even if an extremely small k1
            # makes the floating-point denominator underflow to zero.
            contribution = term["weight"] * term["idf"] * numerator / denominator if tf else 0.0
            details.append({**term, "tf": tf, "matched": tf > 0, "numerator": numerator,
                            "denominator": denominator, "contribution": contribution})
        score = sum(t["contribution"] for t in details)
        candidates.append({"paragraph_id": p["id"], "source_start": p["source_start"],
                           "dl": p["length"], "length_normalization": norm,
                           "term_details": details, "score": score})
    candidates.sort(key=lambda p: (-p["score"], p["source_start"], p["paragraph_id"]))
    for rank, p in enumerate(candidates, start=1):
        p["rank"] = rank
    result = {"query": query, "query_cleaning": query_cleaning, "query_tokens": tokens,
            "query_skipped": skipped, "query_terms": query_terms, "config": asdict(config),
            "formula": FORMULA, "corpus": {"N": N, "avgdl": avgdl, "index_id": index["index_id"]},
            "candidate_policy": "所有查询词倒排列表的并集；无任何命中的段落为零分，不进入结果。",
            "candidate_count": len(candidates), "candidates": candidates,
            "top_ids": [p["paragraph_id"] for p in candidates[:config.top_k]],
            "status": "empty_query" if not tokens else "ok" if candidates else "no_matches"}
    if not legacy:
        result["query_tokenization"] = {
            "policy": CURRENT_QUERY_POLICY if current else LEGACY_ALIAS_QUERY_POLICY if alias_legacy else QUERY_POLICY,
            "rules": {**QUERY_RULES, **(ALIAS_RULES if alias_legacy else CURRENT_ALIAS_RULES if current else {}),
                      **(PERSON_RULES if current else {})},
            "frequency_source": "index.statistics.collection_tf",
            "book_vocabulary_sha256": digest(stats["collection_tf"]),
            "dictionary_sha256": index["dictionary"]["sha256"],
            "candidate_order": ["book_tf descending", "length descending", "term ascending"],
        }
    if has_aliases:
        result["query_expansions"] = expansions
        result["candidate_policy"] = "问题原词及显式别名词项的倒排列表并集；无任何词项命中的段落为零分，不进入结果。"
        result["query_tokenization"].update(alias_sha256=index["aliases"]["sha256"],
            protected_name_order=["name length descending", "name ascending"],
            protected_name_priority="before book-frequency matching")
    if current:
        discovery = index["name_discovery"]
        names = {token["term"] for token in tokens if token["rule"] == "N001"}
        result["query_name_decisions"] = [rec for rec in discovery["candidates"] if rec["accepted"] and rec["name"] in names]
        result["query_ambiguities"] = [item for item in discovery["ambiguities"]
                                      if item["short"] in query_counts or names.intersection(item["full_candidates"])]
        result["candidate_policy"] = "问题原词及本书已接受的同一人物关系扩展词项的倒排列表并集；歧义或证据不足的关系不自动扩展。"
        result["query_tokenization"].update(person_discovery_sha256=discovery["sha256"],
            name_protection_rule="N001", manual_groups=index["aliases"]["manual_groups"])
    return result
