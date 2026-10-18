"""Names remain whole; explicit aliases broaden evidence without duplicate scores."""

import copy
from pathlib import Path

import pytest

from evidence import ENGINE_VERSION
from evidence.dag_tokenization import lexicon_snapshot
from evidence.retrieval import build_index, digest, prepare_index_for_search, search
from evidence.tracing import deserialize, make_trace, read_record, replay, serialize


ROOT = Path(__file__).resolve().parents[1]


def corpus(text, words=None, **kwargs):
    kwargs.setdefault("alias_groups", [["林黛玉", "黛玉"]])
    kwargs.setdefault("lexicon", lexicon_snapshot({}))
    return build_index(text.encode("utf-8"), "人名回归.txt", "utf-8", words or [], **kwargs)


def terms(result):
    return {item["term"]: item for item in result["query_terms"]}


def candidates(result):
    return {item["paragraph_id"]: item for item in result["candidates"]}


def resign(trace):
    trace["integrity_sha256"] = digest({k: v for k, v in trace.items() if k != "integrity_sha256"})


@pytest.mark.parametrize("query", ["黛玉", "林黛玉"])
def test_manual_names_are_whole_without_base_dictionary_entries(query):
    index = corpus("黛玉。\n林黛玉。\n王熙凤。")
    result = search(index, query)

    assert [item["term"] for item in result["query_tokens"]] == [query]
    assert index["paragraphs"][0]["term_frequencies"] == {"黛玉": 1}
    assert index["paragraphs"][1]["term_frequencies"] == {"林黛玉": 1}
    assert {"林黛玉", "黛玉"}.issubset(index["dictionary"]["words"])
    assert set(terms(result)) == {"黛玉", "林黛玉"}
    assert set(result["top_ids"]) == {"P000001", "P000002"}
    assert result["candidate_count"] == 2
    assert result["query_expansions"]
    assert terms(result)[query]["query_frequency"] == 1
    other_name = "林黛玉" if query == "黛玉" else "黛玉"
    assert terms(result)[other_name]["query_frequency"] == 0


def test_names_override_higher_frequency_prefix_but_keep_original_positions():
    index = corpus("林黛。\n" * 12 + "林黛玉。\n黛玉。", words=["林黛"])
    query = "\ufeff　林黛玉\t黛玉？"
    result = search(index, query)

    assert index["statistics"]["collection_tf"]["林黛"] == 12
    assert [item["term"] for item in result["query_tokens"]] == ["林黛玉", "黛玉"]
    for item in result["query_tokens"]:
        assert query[item["source_start"]:item["source_end"]] == item["surface"]
    assert set(result["top_ids"]) == {"P000013", "P000014"}
    assert "林黛" not in terms(result)


def test_alias_expansion_uses_literal_postings_and_records_missing_tf_as_zero():
    index = corpus("黛玉黛玉。\n林黛玉。\n林黛玉黛玉。\n贾宝玉。", words=["贾宝玉"])
    result = search(index, "林黛玉")
    rows = candidates(result)

    assert set(rows) == {"P000001", "P000002", "P000003"}
    assert set(index["postings"]["林黛玉"]) == {"P000002", "P000003"}
    assert set(index["postings"]["黛玉"]) == {"P000001", "P000003"}
    assert index["statistics"]["collection_tf"]["林黛玉"] == 2
    assert index["statistics"]["collection_tf"]["黛玉"] == 3
    for pid, expected in {
        "P000001": {"林黛玉": 0, "黛玉": 2},
        "P000002": {"林黛玉": 1, "黛玉": 0},
        "P000003": {"林黛玉": 1, "黛玉": 1},
    }.items():
        row = rows[pid]
        details = {item["term"]: item for item in row["term_details"]}
        assert {term: item["tf"] for term, item in details.items()} == expected
        for term, tf in expected.items():
            assert details[term]["matched"] is bool(tf)
            if not tf:
                assert details[term]["contribution"] == 0
        assert row["score"] == sum(item["contribution"] for item in row["term_details"])


def test_repeated_and_explicit_aliases_do_not_duplicate_bm25_contributions():
    index = corpus("黛玉。\n林黛玉。\n林黛玉黛玉。")
    baseline = search(index, "林黛玉")
    repeated = search(index, "林黛玉 黛玉 林黛玉")
    query_terms = terms(repeated)

    assert len(repeated["query_terms"]) == 2
    assert query_terms["林黛玉"]["query_frequency"] == 2
    assert query_terms["黛玉"]["query_frequency"] == 1
    assert all(item["weight"] == 1 for item in repeated["query_terms"])
    assert repeated["top_ids"] == baseline["top_ids"]
    assert {pid: row["score"] for pid, row in candidates(repeated).items()} == {
        pid: row["score"] for pid, row in candidates(baseline).items()
    }
    assert all(len(row["term_details"]) == 2 for row in repeated["candidates"])


def test_alias_provenance_distinguishes_literal_query_terms_from_expanded_terms():
    index = corpus("黛玉。\n林黛玉。")
    result = search(index, "林黛玉")
    query_terms = terms(result)

    assert result["query_tokens"][0]["rule"] == "N001"
    assert result["query_expansions"] == [{
        "source_term": "林黛玉", "term": "黛玉", "group": ["林黛玉", "黛玉"],
        "group_id": "G000001", "rule": "A002", "weight": 1, "added_to_query": True,
        "origin": "manual", "relations": [],
    }]
    assert query_terms["林黛玉"]["sources"] == [
        {"kind": "literal", "term": "林黛玉", "query_frequency": 1},
    ]
    assert query_terms["黛玉"]["sources"] == [
        {"kind": "alias", "from": "林黛玉", "rule": "A002", "group_id": "G000001", "origin": "manual"},
    ]
    both = search(index, "林黛玉 黛玉")
    assert all(not item["added_to_query"] for item in both["query_expansions"])
    for term, detail in terms(both).items():
        assert detail["query_frequency"] == 1
        assert {source["kind"] for source in detail["sources"]} == {"literal", "alias"}


@pytest.mark.parametrize("query, paragraph_id", [("黛", "P000004"), ("玉", "P000005")])
def test_name_substrings_do_not_guess_aliases_or_match_other_people(query, paragraph_id):
    index = corpus("黛玉。\n林黛玉。\n贾宝玉。\n黛。\n玉。", words=["贾宝玉"])
    result = search(index, query)

    assert set(terms(result)) == {query}
    assert result["query_expansions"] == []
    assert result["top_ids"] == [paragraph_id]


def test_explicitly_disabled_aliases_keep_literal_matching():
    index = corpus("黛玉。\n林黛玉。", words=["林黛玉", "黛玉"], alias_groups=[])
    result = search(index, "林黛玉")

    assert index["aliases"]["groups"] == []
    assert set(terms(result)) == {"林黛玉"}
    assert result["query_expansions"] == []
    assert result["top_ids"] == ["P000002"]
    assert replay(make_trace(index, result))["verified"]


def test_custom_alias_groups_are_versioned_normalized_and_do_not_add_default_group():
    text = "林姑娘。\n潇湘妃子。\n黛玉。\n林黛玉。"
    a = corpus(text, alias_groups=[["林姑娘", "潇湘妃子"]])
    b = corpus(text, alias_groups=[["潇湘妃子", "林姑娘", "林姑娘"]])
    disabled = corpus(text, alias_groups=[])

    assert a == b
    assert a["index_id"] != disabled["index_id"]
    assert a["aliases"]["groups"] == [sorted(["林姑娘", "潇湘妃子"])]
    assert set(terms(search(a, "林姑娘"))) == {"林姑娘", "潇湘妃子"}
    assert set(search(a, "林姑娘")["top_ids"]) == {"P000001", "P000002"}
    assert "林黛玉" not in terms(search(a, "黛玉"))
    assert replay(make_trace(a, search(a, "林姑娘")))["verified"]


def test_separate_alias_groups_never_cross_expand_by_shared_name_suffix():
    index = corpus("黛玉。\n林黛玉。\n宝玉。\n贾宝玉。", alias_groups=[
        ["贾宝玉", "宝玉"], ["林黛玉", "黛玉"],
    ])
    reversed_groups = corpus("黛玉。\n林黛玉。\n宝玉。\n贾宝玉。", alias_groups=[
        ["黛玉", "林黛玉"], ["宝玉", "贾宝玉"],
    ])

    assert index == reversed_groups
    assert set(terms(search(index, "林黛玉"))) == {"林黛玉", "黛玉"}
    assert set(search(index, "林黛玉")["top_ids"]) == {"P000001", "P000002"}
    assert set(terms(search(index, "宝玉"))) == {"宝玉", "贾宝玉"}
    assert set(search(index, "宝玉")["top_ids"]) == {"P000003", "P000004"}


@pytest.mark.parametrize("groups", [
    "林黛玉 黛玉", {"林黛玉": "黛玉"}, ["林黛玉", "黛玉"],
    [["黛玉"]], [["林黛玉", "黛玉"], ["黛玉", "潇湘妃子"]],
    [["黛玉", "玉"]], [["黛玉", "林 黛玉"]], [["黛玉", 12]],
])
def test_invalid_or_ambiguous_alias_groups_are_rejected(groups):
    with pytest.raises(ValueError):
        corpus("黛玉。", alias_groups=groups)


def test_export_roundtrip_and_recomputation_verify_alias_decisions():
    index = corpus("黛玉。\n林黛玉。\n薛宝钗。", words=["薛宝钗"])
    result = search(index, "林黛玉")
    trace = make_trace(index, result)

    for compress in (False, True):
        report = replay(deserialize(serialize(trace, compress)))
        assert report["verified"]
        assert report["top_ids"] == result["top_ids"]
    tampered = copy.deepcopy(trace)
    tampered["retrieval"]["query_expansions"] = []
    resign(tampered)
    with pytest.raises(ValueError, match="query_expansions"):
        replay(tampered)
    tampered = copy.deepcopy(trace)
    tampered["index"]["aliases"]["groups"] = []
    resign(tampered)
    with pytest.raises(ValueError):
        replay(tampered)


def test_old_index_is_upgraded_without_mutation_before_alias_search_and_export():
    old_index = corpus("黛玉点头。\n林黛玉说道：好。", _engine_version="evidence-bm25/1.1.0")
    snapshot = copy.deepcopy(old_index)
    upgraded = prepare_index_for_search(old_index)

    assert old_index == snapshot
    assert upgraded["identity"]["engine_version"] == ENGINE_VERSION
    assert upgraded["index_id"] != old_index["index_id"]
    assert upgraded["source"] == old_index["source"]
    assert upgraded["paragraphs"][0]["term_frequencies"]["黛玉"] == 1
    result = search(old_index, "林黛玉")
    assert result["corpus"]["index_id"] == upgraded["index_id"]
    assert set(result["top_ids"]) == {"P000001", "P000002"}
    assert replay(make_trace(upgraded, result))["verified"]


def test_frozen_frequency_query_on_legacy_index_keeps_historical_policy_and_ranking():
    frozen = read_record(ROOT / "tests" / "fixtures" / "frequency-trace.json")
    snapshot = copy.deepcopy(frozen)

    assert frozen["engine_version"] == "evidence-bm25/1.1.0"
    assert frozen["index"]["identity"]["engine_version"] == "evidence-bm25/1.0.0"
    report = replay(frozen)
    assert report["verified"]
    assert report["query_policy"] == "book-frequency/1"
    assert report["top_ids"] == ["P000001", "P000002", "P000003"]
    assert frozen == snapshot
    assert "query_expansions" not in frozen["retrieval"]

    reexported = make_trace(frozen["index"], frozen["retrieval"])
    assert reexported["engine_version"] == "evidence-bm25/1.1.0"
    assert reexported["schema"] == "evidence-trace/2"
    assert reexported["index"] == frozen["index"]
    assert reexported["retrieval"] == frozen["retrieval"]
    assert replay(reexported)["query_policy"] == "book-frequency/1"
