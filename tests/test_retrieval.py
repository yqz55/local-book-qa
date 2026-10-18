import copy
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import pytest

from evidence.ingest import IngestConfig, SourceMap, clean_text, decode_txt
from evidence.retrieval import BM25Config, build_index, digest, search
from evidence.storage import Store
from evidence.tokenization import Tokenizer
from evidence.dag_tokenization import lexicon_snapshot
from evidence.tracing import deserialize, make_trace, replay, serialize

ROOT = Path(__file__).resolve().parents[1]


def index(text, words=None, max_chars=400):
    return build_index(text.encode("utf-8"), "测试.txt", "utf-8", words or [], IngestConfig(max_chars), lexicon=lexicon_snapshot({}))


def test_hand_calculated_bm25_and_missing_term():
    # A: 甲甲乙 (dl=3), B: 甲丙 (dl=2), C: 乙 (dl=1).
    # N=3, avgdl=2, df(甲)=2, idf=ln(1.6), k1=1.5, b=.75.
    corpus = index("甲甲乙\n甲丙\n乙")
    result = search(corpus, "甲丁")
    assert corpus["statistics"]["avgdl"] == 2
    assert corpus["statistics"]["collection_tf"]["甲"] == 3
    assert corpus["statistics"]["document_frequency"]["甲"] == 2
    a, b = result["candidates"]
    assert a["paragraph_id"] == "P000001"
    assert a["score"] == pytest.approx(math.log(1.6) * 5 / 4.0625)
    assert b["score"] == pytest.approx(math.log(1.6))
    assert a["term_details"][1]["tf"] == 0
    assert a["term_details"][1]["idf"] == pytest.approx(math.log(8))
    assert a["term_details"][1]["contribution"] == 0
    assert a["score"] == sum(t["contribution"] for t in a["term_details"])


def test_source_offsets_cleaning_chapters_and_unicode():
    raw = "\ufeff　第一章 测试  \r\n\r\n  林黛玉\t来😀。　\r尾段\n"
    corpus = index(raw, ["林黛玉"])
    cleaned = corpus["cleaning"]
    assert cleaned["text"] == "第一章 测试\n\n林黛玉 来😀。\n尾段\n"
    p = corpus["paragraphs"][1]
    assert p["location"]["start_line"] == 3
    assert p["location"]["start_column"] == 3
    assert raw[p["source_start"]:p["source_end"]] == "林黛玉\t来😀。"
    assert p["chapter"] == "第一章 测试"
    assert corpus["paragraphs"][0]["split_rule"] == "S003"
    token = next(t for t in p["tokens"] if t["term"] == "😀")
    assert raw[token["source_start"]:token["source_end"]] == "😀"
    assert token["source_end"] - token["source_start"] == 1  # not UTF-16 units
    assert {e["rule"] for e in cleaned["edits"]} == {"C001", "C002", "C003", "C004"}
    mapping = SourceMap(cleaned["source_runs"], raw)
    for position, char in enumerate(cleaned["text"]):
        a, b = mapping.span(position, position + 1)
        original = raw[a:b]
        assert original == char or (char == "\n" and original in {"\r", "\r\n"}) or (char == " " and original in {"\t", "\u3000"})


def test_length_splitting_is_lossless_with_sentence_and_hard_fallback():
    text = "甲" * 8 + "。" + "乙" * 30
    corpus = index(text, max_chars=16)
    ps = corpus["paragraphs"]
    assert "".join(p["text"] for p in ps) == text
    assert [p["split_rule"] for p in ps] == ["S004", "S005", "S002"]
    assert all(len(p["text"]) <= 16 for p in ps)
    assert all(a["source_end"] == b["source_start"] for a, b in zip(ps, ps[1:]))


def test_longest_dictionary_matching_and_explicit_fallbacks():
    tokens, skipped = Tokenizer(["信息", "信息检索", "检索"]).tokenize("信息检索未知 ABC-12 3.14 α +！")
    assert [t["term"] for t in tokens] == ["信息检索", "未", "知", "abc-12", "3.14", "α", "+"]
    assert [t["rule"] for t in tokens[:4]] == ["T001", "T002", "T002", "T003"]
    assert any("！" in t["surface"] for t in skipped)
    assert Tokenizer(["信息检索", "信息"]).sha256 == Tokenizer(["信息", "信息检索", "信息"]).sha256


def test_query_duplicate_policy_normalization_and_tie_break_top5():
    corpus = index("\n".join(["甲乙"] * 7))
    result = search(corpus, "\ufeff 甲甲　\r\n")
    single = search(corpus, "甲")
    assert result["top_ids"] == [f"P{i:06}" for i in range(1, 6)]
    assert result["candidate_count"] == 7
    assert result["query_terms"][0]["query_frequency"] == 2
    assert result["query_terms"][0]["weight"] == 1
    assert [p["score"] for p in result["candidates"]] == [p["score"] for p in single["candidates"]]


def test_empty_zero_and_oov_handling():
    corpus = index("甲\n！！！\n乙")
    assert corpus["statistics"]["N"] == 2
    assert corpus["statistics"]["paragraph_count"] == 3
    assert corpus["paragraphs"][1]["indexed"] is False
    assert search(corpus, "？？ \n")["status"] == "empty_query"
    assert search(corpus, "丁")["status"] == "no_matches"
    assert search(corpus, "丁")["top_ids"] == []
    with pytest.raises(ValueError, match="可检索词项"):
        index("！？")


def test_explicit_encoding_no_silent_guessing():
    text = "中文书籍，贾宝玉。"
    data = text.encode("gb18030")
    assert decode_txt(data, "gb18030") == text
    corpus = build_index(data, "书.txt", "gb18030", ["贾宝玉"])
    assert replay(make_trace(corpus, search(corpus, "贾宝玉")))["verified"]
    with pytest.raises(ValueError, match="严格解码"):
        decode_txt(data, "utf-8")
    with pytest.raises(ValueError, match="NUL"):
        decode_txt(b"a\0b", "utf-8")
    with pytest.raises(ValueError, match="编码"):
        decode_txt(b"abc", "auto")


@pytest.mark.parametrize("config", [dict(k1=0), dict(k1=float("nan")), dict(k1=float("inf")),
                                   dict(b=-.1), dict(b=1.1), dict(top_k=0), dict(k1=True)])
def test_bad_parameters_rejected(config):
    with pytest.raises(ValueError):
        BM25Config(**config)


def test_versions_invalidate_on_rule_dictionary_or_bytes_change():
    a = index("信息检索", ["信息"])
    assert a["index_id"] != index("信息检索", ["信息检索"])["index_id"]
    assert a["index_id"] != index("信息检索\n", ["信息"])["index_id"]
    assert a["index_id"] != index("信息检索", ["信息"], max_chars=32)["index_id"]


def test_replay_full_rebuild_and_two_levels_of_tamper_detection():
    corpus = index("\ufeff甲甲乙\r\n甲丙\r\n乙")
    original = make_trace(corpus, search(corpus, "甲乙"))
    for compress in (False, True):
        report = replay(deserialize(serialize(original, compress)))
        assert report["verified"] is True
        assert report["top_ids"] == original["retrieval"]["top_ids"]
    tampered = copy.deepcopy(original)
    tampered["retrieval"]["candidates"][0]["score"] += .1
    with pytest.raises(ValueError, match="完整性"):
        replay(tampered)
    tampered["integrity_sha256"] = digest({k: v for k, v in tampered.items() if k != "integrity_sha256"})
    with pytest.raises(ValueError, match="复算不一致.*score"):
        replay(tampered)
    tampered = copy.deepcopy(original)
    tampered["index"]["paragraphs"][0]["tokens"][0]["source_start"] = 999
    tampered["integrity_sha256"] = digest({k: v for k, v in tampered.items() if k != "integrity_sha256"})
    with pytest.raises(ValueError, match="source_start"):
        replay(tampered)


def test_snapshot_survives_external_dictionary_and_source_removal(tmp_path):
    corpus = index("信息检索\n其他", ["信息检索"])
    store = Store(tmp_path)
    path = store.save_index(corpus)
    trace = make_trace(corpus, search(corpus, "信息检索"))
    trace_path = store.save_trace(trace)
    assert len(store.list_indexes()) == 1
    assert Path(store.list_indexes()[0]["path"]) == path
    assert Path(store.list_traces()[0]["path"]) == trace_path
    # A new interpreter only needs its trace, not the store or current dictionary.
    trace_copy = tmp_path / "standalone.json.gz"
    trace_copy.write_bytes(trace_path.read_bytes())
    result = subprocess.run([sys.executable, "-m", "evidence.cli", "replay", str(trace_copy)],
                            cwd=ROOT, capture_output=True, encoding="utf-8", env={**__import__('os').environ, "PYTHONIOENCODING": "utf-8"})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["top_ids"] == ["P000001"]


def test_randomized_clean_map_and_deterministic_ranking():
    rng = random.Random(42)
    for _ in range(20):
        text = "\ufeff" + "".join(rng.choice(["甲", "乙", "😀", "，", "\r\n", "\r", "\n", "　", "\t", "a"]) for _ in range(120))
        corpus = index(text, max_chars=16)
        mapping = SourceMap(corpus["cleaning"]["source_runs"], text)
        for p in corpus["paragraphs"]:
            assert mapping.span(p["clean_start"], p["clean_end"]) == (p["source_start"], p["source_end"])
            for t in p["tokens"]:
                assert text[t["source_start"]:t["source_end"]] == t["surface"]
        a, b = search(corpus, "甲乙"), search(corpus, "甲乙")
        assert a == b
        assert replay(make_trace(corpus, a))["verified"]


def test_invalid_dictionary_and_json_are_rejected():
    for words in (["a"], ["甲"], ["甲 乙"], "信息", [12]):
        with pytest.raises(ValueError):
            Tokenizer(words)
    with pytest.raises(ValueError, match="重复字段"):
        deserialize(b'{"schema":1,"schema":2}')
    with pytest.raises(ValueError, match="非有限"):
        deserialize(b'{"x": NaN}')


def test_extremely_small_k1_and_invalid_snapshot_types():
    corpus = index("甲\n乙乙乙乙乙乙")
    result = search(corpus, "甲丁", BM25Config(k1=5e-324, b=1))
    assert result["candidates"][0]["term_details"][1]["contribution"] == 0.0
    assert replay(make_trace(corpus, result))["verified"]
    with pytest.raises(ValueError, match="格式"):
        replay([])
