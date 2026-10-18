"""Book-frequency decisions, dictionary edits, and frozen v1 compatibility."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from evidence import PERSON_ENGINE_VERSION as ENGINE_VERSION, LEGACY_ENGINE_VERSION
from evidence.legacy_retrieval import build_index, digest, rebuild_with_book_words, search
from evidence.storage import Store
from evidence.tokenization import Tokenizer
from evidence.tracing import make_trace, read_record, replay

ROOT = Path(__file__).resolve().parents[1]
WORDS = ["信息", "信息检索", "检索"]


def corpus(text, words=WORDS, book_words=None):
    return build_index(text.encode("utf-8"), "词表测试.txt", "utf-8", list(words), book_words=book_words)


def resign(trace):
    trace["integrity_sha256"] = digest({k: v for k, v in trace.items() if k != "integrity_sha256"})


def test_book_frequency_can_choose_shorter_than_longest():
    index = corpus("信息\n" * 5 + "信息检索\n" + "检索\n" * 3)
    result = search(index, "信息检索")
    assert [t["term"] for t in result["query_tokens"]] == ["信息", "检索"]
    first = result["query_tokens"][0]
    assert first["match_candidates"] == [
        {"term": "信息", "book_tf": 5, "length": 2},
        {"term": "信息检索", "book_tf": 1, "length": 4},
    ]
    assert first["match_source"] == "book_vocabulary"
    assert first["book_frequency"] == 5
    assert first["selection_rule"] == "Q002"
    assert result["candidate_count"] == 8
    assert all(t["weight"] == 1 for t in result["query_terms"])
    assert result["query_tokenization"]["book_vocabulary_sha256"] == digest(index["statistics"]["collection_tf"])
    assert replay(make_trace(index, result))["verified"]


def test_frequency_ties_choose_longer_and_single_characters_are_fallback():
    index = corpus("信息\n信息检索\n检索")
    result = search(index, "信息检索")
    assert result["query_tokens"][0]["term"] == "信息检索"
    assert [c["term"] for c in result["query_tokens"][0]["match_candidates"]] == ["信息检索", "信息"]
    index = corpus("信\n" * 20 + "信息检索")
    first = search(index, "信息检索")["query_tokens"][0]
    assert index["statistics"]["collection_tf"]["信"] == 20
    assert first["term"] == "信息检索"
    assert [c["term"] for c in first["match_candidates"]] == ["信息检索"]


def test_frequency_means_segmented_terms_not_arbitrary_substrings():
    index = corpus("信息检索\n信息检索")
    assert "信息" not in index["statistics"]["collection_tf"]
    whole = search(index, "信息检索")["query_tokens"][0]
    assert whole["match_candidates"] == [{"term": "信息检索", "book_tf": 2, "length": 4}]
    absent = search(index, "信息")["query_tokens"][0]
    assert absent["match_source"] == "saved_dictionary"
    assert absent["book_frequency"] == 0
    assert absent["match_candidates"] == []
    assert absent["selection_rule"] == "Q003"


def test_query_fallbacks_and_original_coordinates_are_preserved():
    index = corpus("信息检索 ABC-12 α +")
    query = "\ufeff　信息检索\tABC-12 未知 α +！"
    result = search(index, query)
    tokens = result["query_tokens"]
    assert [t["term"] for t in tokens] == ["信息检索", "abc-12", "未", "知", "α", "+"]
    assert [t["match_source"] for t in tokens] == ["book_vocabulary", "ascii_run",
           "han_character_fallback", "han_character_fallback", "unicode_fallback", "unicode_fallback"]
    for token in tokens:
        assert query[token["source_start"]:token["source_end"]] == token["surface"]
    assert result["status"] == "ok"
    assert replay(make_trace(index, result))["verified"]


def test_dictionary_edit_rebuilds_document_query_tf_and_versions_together():
    original = corpus("光合作用。\n光合作用。", words=[])
    original_snapshot = copy.deepcopy(original)
    old_trace = make_trace(original, search(original, "光合作用"))
    updated = rebuild_with_book_words(original, ["光合作用"])
    assert original == original_snapshot
    assert updated["source"] == original["source"]
    assert updated["index_id"] != original["index_id"]
    assert updated["dictionary"]["base_words"] == []
    assert updated["dictionary"]["book_words"] == ["光合作用"]
    assert updated["statistics"]["collection_tf"]["光合作用"] == 2
    assert updated["statistics"]["total_tokens"] == 2
    assert original["statistics"]["total_tokens"] == 8
    result = search(updated, "光合作用")
    assert [t["term"] for t in result["query_tokens"]] == ["光合作用"]
    assert result["query_tokens"][0]["book_frequency"] == 2
    assert replay(old_trace)["verified"]
    assert replay(make_trace(updated, result))["verified"]
    removed = rebuild_with_book_words(updated, [])
    assert removed == original


def test_book_words_merge_with_base_and_removal_does_not_erase_base():
    original = corpus("信息检索与光合作用。", book_words=["光合作用"])
    assert set(WORDS).issubset(original["dictionary"]["words"])
    assert "光合作用" in original["dictionary"]["words"]
    removed = rebuild_with_book_words(original, [])
    assert removed["dictionary"]["base_words"] == sorted(WORDS)
    assert set(removed["dictionary"]["words"]) == set(WORDS)
    assert "光合作用" not in removed["statistics"]["collection_tf"]


def test_frozen_v1_record_keeps_original_ranking_and_old_index_supports_new_queries():
    frozen = read_record(ROOT / "tests" / "fixtures" / "legacy-trace.json")
    assert frozen["engine_version"] == LEGACY_ENGINE_VERSION
    previous = replay(frozen)
    assert previous["top_ids"] == ["P000004"]
    assert previous["query_policy"] == "base-longest/legacy-1.0"
    # New queries create a current index; the historical snapshot remains intact.
    result = search(frozen["index"], "机器学习")
    assert [t["term"] for t in result["query_tokens"]] == ["机器", "学习"]
    assert result["top_ids"] == ["P000001", "P000002", "P000003"]
    new_record = make_trace(frozen["index"], result)
    assert new_record["engine_version"] == ENGINE_VERSION
    assert new_record["index"]["identity"]["engine_version"] == ENGINE_VERSION
    assert new_record["index"]["source"] == frozen["index"]["source"]
    assert frozen["index"]["identity"]["engine_version"] == LEGACY_ENGINE_VERSION
    assert replay(new_record)["query_policy"] == "book-person-rules/3"
    upgraded = rebuild_with_book_words(frozen["index"], ["新词"])
    assert upgraded["identity"]["engine_version"] == ENGINE_VERSION
    assert upgraded["dictionary"]["base_words"] == frozen["index"]["dictionary"]["words"]


def test_tampered_query_choice_detected_even_with_recomputed_integrity_hash():
    index = corpus("信息\n" * 4 + "信息检索")
    trace = make_trace(index, search(index, "信息检索"))
    trace["retrieval"]["query_tokens"][0]["match_candidates"][0]["book_tf"] += 1
    resign(trace)
    with pytest.raises(ValueError, match="book_tf"):
        replay(trace)


def test_dictionary_components_and_iteration_order_have_stable_versions():
    a = corpus("信息检索与光合作用。", book_words=["光合作用", "新词"])
    b = corpus("信息检索与光合作用。", words=list(reversed(WORDS)), book_words=["新词", "光合作用", "新词"])
    assert a == b
    assert search(a, "信息检索光合作用") == search(b, "信息检索光合作用")
    # Provenance changes are versioned even when the effective union is identical.
    c = corpus("信息检索", book_words=["信息检索"])
    d = corpus("信息检索")
    assert c["dictionary"]["words"] == d["dictionary"]["words"]
    assert c["index_id"] != d["index_id"]


@pytest.mark.parametrize("book_words", [["a"], ["甲"], ["甲 乙"], [12], {"词": 1}])
def test_invalid_book_words_rejected(book_words):
    with pytest.raises(ValueError):
        corpus("信息检索", book_words=book_words)


def test_cli_rebuild_and_standalone_replay(tmp_path):
    original = corpus("光合作用。", words=[])
    original_path = Store(tmp_path).save_index(original)
    original_bytes = original_path.read_bytes()
    words_path = tmp_path / "book-words.json"
    words_path.write_text('["光合作用"]', encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    rebuilt = subprocess.run([sys.executable, "-m", "evidence.cli", "rebuild", str(original_path),
                             "--book-dictionary", str(words_path), "--data-dir", str(tmp_path)],
                             cwd=ROOT, capture_output=True, encoding="utf-8", env=env)
    assert rebuilt.returncode == 0, rebuilt.stderr
    new_path = Path(json.loads(rebuilt.stdout)["index"])
    trace_path = tmp_path / "replay.json.gz"
    queried = subprocess.run([sys.executable, "-m", "evidence.cli", "search", str(new_path),
                              "光合作用", "--export", str(trace_path)],
                              cwd=ROOT, capture_output=True, encoding="utf-8", env=env)
    assert queried.returncode == 0, queried.stderr
    assert json.loads(queried.stdout)["query_tokens"][0]["book_frequency"] == 1
    replayed = subprocess.run([sys.executable, "-m", "evidence.cli", "replay", str(trace_path)],
                              cwd=ROOT, capture_output=True, encoding="utf-8", env=env)
    assert replayed.returncode == 0, replayed.stderr
    assert json.loads(replayed.stdout)["verified"]
    assert original_path.read_bytes() == original_bytes
