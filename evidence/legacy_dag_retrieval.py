"""Two-pass dictionary DAG and BM25, with frozen historical dispatch."""

import base64
from collections import Counter
from dataclasses import asdict
import hashlib
import math
import unicodedata

from . import DAG_ENGINE_VERSION as ENGINE_VERSION, LEGACY_ENGINE_VERSION, SUPPORTED_ENGINE_VERSIONS, schema_version
from . import legacy_retrieval
from .aliases import CURRENT_ALIAS_RULES, alias_words, expand_query, inferred_alias_snapshot
from .dag_tokenization import DAG_POLICY, DAG_RULES, DAGTokenizer, default_lexicon, lexicon_snapshot
from .ingest import IngestConfig, RULES, SourceMap, clean_text, ingest
from .legacy_retrieval import BM25Config, FORMULA, canonical, digest, _manual_groups
from .person_names import PERSON_RULES, discover_people
from .tokenization import TOKEN_RULES, Tokenizer

SEGMENTATION_CONFIG = {"policy": DAG_POLICY, "hmm": False, "book_bonus_cap": 0.25,
    "adaptation_passes": 1, "tie_break": "path score descending, length descending, term ascending"}


def build_index(data, filename, encoding, words, config=None, *, book_words=None, alias_groups=None,
                person_config=None, lexicon=None, segmentation_config=None, _engine_version=ENGINE_VERSION):
    if _engine_version != ENGINE_VERSION:
        if _engine_version not in SUPPORTED_ENGINE_VERSIONS:
            raise ValueError("不支持的索引引擎版本。")
        return legacy_retrieval.build_index(data, filename, encoding, words, config, book_words=book_words,
            alias_groups=alias_groups, person_config=person_config, _engine_version=_engine_version)
    config = config or IngestConfig()
    cfg = dict(SEGMENTATION_CONFIG if segmentation_config is None else segmentation_config)
    if (cfg.keys() != SEGMENTATION_CONFIG.keys() or cfg["policy"] != DAG_POLICY or cfg["hmm"] is not False
            or cfg["adaptation_passes"] != 1 or cfg["tie_break"] != SEGMENTATION_CONFIG["tie_break"]):
        raise ValueError("不支持的整句分词配置。")
    resource = default_lexicon() if lexicon is None else lexicon
    checked = lexicon_snapshot(resource["frequencies"], source=resource["source"], total=resource["total"],
                               resource_sha256=resource["resource_sha256"])
    if checked != resource:
        raise ValueError("通用词典快照校验失败。")
    base, additions = Tokenizer(words), Tokenizer([] if book_words is None else book_words)
    raw, cleaned, paragraphs = ingest(data, encoding, config)
    discovery = discover_people(cleaned, raw, person_config)
    aliases = inferred_alias_snapshot(discovery, alias_groups)
    names = sorted(set(discovery["names"]) | set(alias_words(aliases)))
    extras = sorted(set(base.words) | set(additions.words) | set(names))
    tokenizer = DAGTokenizer(resource, extras, names, additions.words, book_bonus_cap=cfg["book_bonus_cap"])
    mapping, seed_tf = SourceMap(cleaned["source_runs"], raw), Counter()
    for p in paragraphs:
        tokens, _ = tokenizer.tokenize(p["text"])
        seed_tf.update(t["term"] for t in tokens)
    seed_tf = dict(sorted(seed_tf.items()))
    tokenizer.set_book_frequencies(seed_tf)
    postings, collection_tf, df, lengths = {}, Counter(), Counter(), []
    for p in paragraphs:
        tokens, skipped = tokenizer.tokenize(p["text"], p["clean_start"], mapping)
        tf = dict(sorted(Counter(t["term"] for t in tokens).items()))
        p.update(tokens=tokens, skipped=skipped, term_frequencies=tf, length=len(tokens), indexed=bool(tokens))
        if tokens:
            lengths.append(len(tokens))
        for term, count in tf.items():
            postings.setdefault(term, []).append(p["id"])
            collection_tf[term] += count
            df[term] += 1
    if not lengths:
        raise ValueError("文本没有可检索词项（可能只含标点或空白）。")
    for p in paragraphs:
        for token in p["tokens"]:
            token["book_frequency"] = collection_tf[token["term"]]
    segmentation = {"policy": DAG_POLICY, "implementation": "audited jieba dictionary DAG / HMM=False",
        "lexicon": resource, "config": cfg, "seed_collection_tf": seed_tf,
        "seed_vocabulary_sha256": digest(seed_tf), "rules": DAG_RULES}
    segmentation["sha256"] = digest(segmentation)
    dictionary = {"words": tokenizer.words, "sha256": tokenizer.sha256, "base_words": base.words,
        "book_words": additions.words, "general_lexicon_sha256": resource["sha256"]}
    identity = {"engine_version": ENGINE_VERSION, "bytes_sha256": hashlib.sha256(data).hexdigest(),
        "encoding": encoding, "ingest_config": asdict(config), "dictionary_sha256": tokenizer.sha256,
        "base_dictionary_sha256": base.sha256, "book_dictionary_sha256": additions.sha256,
        "alias_sha256": aliases["sha256"], "person_discovery_sha256": discovery["sha256"],
        "segmentation_sha256": segmentation["sha256"], "unicode_version": unicodedata.unidata_version}
    return {"schema": f"evidence-index/{schema_version(ENGINE_VERSION)}", "index_id": digest(identity), "identity": identity,
        "source": {"filename": filename, "encoding": encoding, "byte_length": len(data),
            "bytes_base64": base64.b64encode(data).decode("ascii"), "text": raw,
            "text_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()},
        "dictionary": dictionary, "segmentation": segmentation, "aliases": aliases, "name_discovery": discovery,
        "rules": {**RULES, **{k: TOKEN_RULES[k] for k in ("T003", "T004", "T005")}, **PERSON_RULES, **CURRENT_ALIAS_RULES, **DAG_RULES},
        "cleaning": cleaned, "paragraphs": paragraphs, "postings": dict(sorted(postings.items())),
        "statistics": {"N": len(lengths), "paragraph_count": len(paragraphs),
            "excluded_empty_token_paragraphs": len(paragraphs) - len(lengths), "total_tokens": sum(lengths),
            "avgdl": sum(lengths) / len(lengths), "vocabulary_size": len(df),
            "collection_tf": dict(sorted(collection_tf.items())), "document_frequency": dict(sorted(df.items()))}}


def _rebuild(index, book_words, groups):
    source, dictionary, seg = index["source"], index["dictionary"], index.get("segmentation", {})
    return build_index(base64.b64decode(source["bytes_base64"], validate=True), source["filename"], source["encoding"],
        dictionary.get("base_words", dictionary["words"]), IngestConfig(**index["identity"]["ingest_config"]),
        book_words=book_words, alias_groups=groups, person_config=index.get("name_discovery", {}).get("config"),
        lexicon=seg.get("lexicon"), segmentation_config=seg.get("config"))


def rebuild_with_book_words(index, book_words):
    return _rebuild(index, book_words, _manual_groups(index))


def rebuild_with_aliases(index, groups):
    return _rebuild(index, index["dictionary"].get("book_words", []), groups)


def prepare_index_for_search(index):
    if index["identity"]["engine_version"] == ENGINE_VERSION:
        return index
    if index["identity"]["engine_version"] not in SUPPORTED_ENGINE_VERSIONS:
        raise ValueError("不支持该索引的引擎版本。")
    return rebuild_with_aliases(index, _manual_groups(index))


def search(index, query, config=None, *, _legacy=False, _engine_version=None):
    engine = LEGACY_ENGINE_VERSION if _legacy else _engine_version or ENGINE_VERSION
    if engine != ENGINE_VERSION:
        return legacy_retrieval.search(index, query, config, _engine_version=engine)
    config = config or BM25Config()
    if not isinstance(query, str) or len(query) > 500:
        raise ValueError("问题最多 500 个 Unicode 字符。")
    index = prepare_index_for_search(index)
    stats, seg = index["statistics"], index["segmentation"]
    names = sorted(set(index["name_discovery"]["names"]) | set(alias_words(index["aliases"])))
    extras = sorted(set(index["dictionary"]["base_words"]) | set(index["dictionary"]["book_words"]) | set(names))
    tokenizer = DAGTokenizer(seg["lexicon"], extras, names, index["dictionary"]["book_words"], seg["seed_collection_tf"],
        book_bonus_cap=seg["config"]["book_bonus_cap"], indexed_frequencies=stats["collection_tf"])
    cleaned = clean_text(query)
    tokens, skipped = tokenizer.tokenize(cleaned["text"], mapping=SourceMap(cleaned["source_runs"], query), capture=True)
    counts = Counter(t["term"] for t in tokens)
    sources, expansions = expand_query(counts, index["aliases"])
    terms, ids, N, avgdl = [], set(), stats["N"], stats["avgdl"]
    for term in sources:
        df = stats["document_frequency"].get(term, 0)
        terms.append({"term": term, "query_frequency": counts.get(term, 0), "weight": 1,
            "df": df, "idf": math.log1p((N - df + 0.5) / (df + 0.5)), "in_vocabulary": df > 0, "sources": sources[term]})
        ids.update(index["postings"].get(term, []))
    candidates = []
    for p in index["paragraphs"]:
        if p["id"] not in ids:
            continue
        norm, details = 1 - config.b + config.b * p["length"] / avgdl, []
        for term in terms:
            tf = p["term_frequencies"].get(term["term"], 0)
            numerator, denominator = tf * (config.k1 + 1), tf + config.k1 * norm
            contribution = term["weight"] * term["idf"] * numerator / denominator if tf else 0.0
            details.append({**term, "tf": tf, "matched": tf > 0, "numerator": numerator,
                "denominator": denominator, "contribution": contribution})
        candidates.append({"paragraph_id": p["id"], "source_start": p["source_start"], "dl": p["length"],
            "length_normalization": norm, "term_details": details, "score": sum(t["contribution"] for t in details)})
    candidates.sort(key=lambda p: (-p["score"], p["source_start"], p["paragraph_id"]))
    for rank, candidate in enumerate(candidates, 1):
        candidate["rank"] = rank
    recognized, discovery = {t["term"] for t in tokens if t["rule"] == "N001"}, index["name_discovery"]
    return {"query": query, "query_cleaning": cleaned, "query_tokens": tokens, "query_skipped": skipped,
        "query_terms": terms, "config": asdict(config), "formula": FORMULA,
        "corpus": {"N": N, "avgdl": avgdl, "index_id": index["index_id"]},
        "candidate_policy": "问题原词及已接受人物关系扩展词的倒排并集；每个去重词项权重为 1，不重复计 TF，不混入字符子串词项。",
        "candidate_count": len(candidates), "candidates": candidates, "top_ids": [c["paragraph_id"] for c in candidates[:config.top_k]],
        "status": "empty_query" if not tokens else "ok" if candidates else "no_matches", "query_expansions": expansions,
        "query_name_decisions": [r for r in discovery["candidates"] if r["accepted"] and r["name"] in recognized],
        "query_ambiguities": [a for a in discovery["ambiguities"] if a["short"] in counts or recognized.intersection(a["full_candidates"])],
        "query_segmentation_trace": tokenizer.last_trace,
        "query_tokenization": {"policy": DAG_POLICY, "rules": {**DAG_RULES, **CURRENT_ALIAS_RULES},
            "frequency_source": "index.segmentation.seed_collection_tf (frozen first pass)",
            "book_vocabulary_sha256": seg["seed_vocabulary_sha256"], "dictionary_sha256": tokenizer.sha256,
            "segmentation_sha256": seg["sha256"], "alias_sha256": index["aliases"]["sha256"],
            "person_discovery_sha256": discovery["sha256"], "hmm": False,
            "book_bonus_cap": seg["config"]["book_bonus_cap"],
            "candidate_order": ["whole path score descending", "length descending", "term ascending"]}}
