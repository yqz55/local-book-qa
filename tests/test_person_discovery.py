"""Corpus rules work without person entries, manual groups, or a book allowlist."""

import copy
import json
from pathlib import Path

import pytest

from evidence import ALIAS_ENGINE_VERSION, ENGINE_VERSION
from evidence.dag_tokenization import lexicon_snapshot
from evidence.person_names import default_config
from evidence.retrieval import (build_index, digest, prepare_index_for_search,
                                rebuild_with_aliases, rebuild_with_book_words, search)
from evidence.tracing import make_trace, read_record, replay


def corpus(text, **kwargs):
    return build_index(text.encode("utf-8"), "未见姓名.txt", "utf-8", [], lexicon=lexicon_snapshot({}), **kwargs)


def query_terms(index, query):
    return {t["term"] for t in search(index, query)["query_terms"]}


@pytest.mark.parametrize("surname,given", [
    ("赵", "清岚"), ("陈", "清砚"), ("韩", "映澜"), ("沈", "疏棠"),
    ("欧阳", "若晨"), ("诸葛", "听澜"), ("司马", "晴川"), ("上官", "知遥"),
])
def test_unseeded_surnames_and_compound_surnames_generalize(surname, given):
    full = surname + given
    index = corpus(f"{full}说道：收到。\n{given}点头。\n{given}起身。")
    assert index["dictionary"]["base_words"] == []
    assert index["dictionary"]["book_words"] == []
    assert index["aliases"]["manual_groups"] == []
    assert index["aliases"]["groups"] == [sorted([full, given])]
    for name in (full, given):
        result = search(index, name)
        assert [t["term"] for t in result["query_tokens"]] == [name]
        assert result["query_tokens"][0]["rule"] == "N001"
        assert query_terms(index, name) == {full, given}
        assert result["candidate_count"] == 3
        assert result["query_expansions"][0]["origin"] == "automatic"
        assert result["query_expansions"][0]["relations"][0]["rule"] == "N201"
        assert result["query_name_decisions"]
    assert index["statistics"]["collection_tf"][given] == 2
    assert index["statistics"]["collection_tf"][full] == 1
    # The embedded suffix is discovery evidence, never duplicate document TF.
    assert index["paragraphs"][0]["term_frequencies"].get(given, 0) == 0


def test_unseeded_names_from_different_people_do_not_cross_expand():
    index = corpus("赵清岚说道：好。\n清岚点头。\n欧阳若晨说道：好。\n若晨起身。")
    assert query_terms(index, "赵清岚") == {"赵清岚", "清岚"}
    assert query_terms(index, "若晨") == {"欧阳若晨", "若晨"}
    assert len(index["aliases"]["groups"]) == 2


def test_person_context_can_protect_a_name_without_surname_or_identity_guess():
    index = corpus("星河说道：好。\n星河点头。\n星河起身。")
    result = search(index, "星河")
    assert [t["term"] for t in result["query_tokens"]] == ["星河"]
    assert result["query_expansions"] == []
    assert index["aliases"]["groups"] == []


def test_shared_short_form_stays_whole_but_does_not_merge_different_full_names():
    index = corpus("张若溪说道：好。\n李若溪起身。\n若溪点头。")
    assert index["aliases"]["groups"] == []
    assert query_terms(index, "张若溪") == {"张若溪"}
    assert query_terms(index, "李若溪") == {"李若溪"}
    result = search(index, "若溪")
    assert [t["term"] for t in result["query_tokens"]] == ["若溪"]
    assert result["candidate_count"] == 1
    assert result["query_ambiguities"][0]["full_candidates"] == ["张若溪", "李若溪"]
    assert result["query_ambiguities"][0]["rule"] == "N202"
    assert replay(make_trace(index, result))["verified"]


@pytest.mark.parametrize("text,full,short", [
    ("他姓陈名清砚。\n清砚说道：好。\n清砚点头。", "陈清砚", "清砚"),
    ("星河说道：好。\n星河点头。\n星河本姓周。", "周星河", "星河"),
    ("赵清岚又名若白。\n若白说道：好。\n若白点头。", "赵清岚", "若白"),
    ("欧阳若晨字晴川。\n晴川说道：好。\n晴川点头。", "欧阳若晨", "晴川"),
])
def test_explicit_identity_statements_are_versioned_evidence(text, full, short):
    index = corpus(text)
    assert query_terms(index, full) == {full, short}
    result = search(index, full)
    assert any(e["rule"] == "N203" for e in result["query_expansions"][0]["relations"])
    if full not in text:
        assert index["statistics"]["collection_tf"].get(full, 0) == 0
        assert not index["postings"].get(full)
        assert all(c["term_details"][0]["tf"] == 0 for c in result["candidates"])
    assert replay(make_trace(index, result))["verified"]


def test_frequency_alone_and_generic_person_grammar_are_not_name_evidence():
    index = corpus("普通文字。\n" + "星河与晨光。\n" * 20 +
                   "数据说明问题。\n大家说道：好。\n大家点头。\n他又说道：好。\n姨娘不知说道：好。")
    assert index["name_discovery"]["names"] == []
    assert index["aliases"]["groups"] == []
    assert [t["term"] for t in search(index, "星河")["query_tokens"]] == ["星", "河"]


def test_ordinary_action_is_not_an_explicit_alias_statement():
    index = corpus("星河说道：好。\n星河点头。\n星河也叫了一回。\n星河知道了，便走。")
    assert index["aliases"]["groups"] == []
    assert "了一回" not in index["name_discovery"]["names"]
    assert "星河知" not in index["name_discovery"]["names"]


def test_explicit_human_introduction_can_override_ordinary_grammar_filters():
    index = corpus("那名男子名叫任我行。\n任我行说道：好。")
    assert [t["term"] for t in search(index, "任我行")["query_tokens"]] == ["任我行"]
    record = next(r for r in index["name_discovery"]["candidates"] if r["name"] == "任我行")
    assert record["rule"] == "N103"


def test_rule_configuration_can_disable_discovery_without_affecting_replay():
    cfg = default_config()
    cfg["enabled"] = False
    index = corpus("赵清岚说道：好。\n清岚点头。", person_config=cfg)
    assert index["name_discovery"]["names"] == []
    assert index["aliases"]["groups"] == []
    assert replay(make_trace(index, search(index, "赵清岚")))["verified"]


def test_one_character_is_not_guessed_as_an_alias_and_names_are_not_global():
    a = corpus("赵清岚说道：好。\n清岚点头。\n岚。")
    assert query_terms(a, "岚") == {"岚"}
    assert search(a, "岚")["top_ids"] == ["P000003"]
    b = corpus("其他普通文字。")
    assert b["name_discovery"]["names"] == []
    assert b["aliases"]["groups"] == []
    assert "赵清岚" not in b["dictionary"]["words"]


def test_raw_evidence_offsets_survive_bom_crlf_and_spacing():
    text = "\ufeff　赵清岚说道：好。\r\n\t清岚点头。\r\n清岚起身。"
    index = corpus(text)
    discovery = index["name_discovery"]
    assert discovery["config"] == default_config()
    assert discovery["counts"]["automatic_groups"] == 1
    for rec in discovery["candidates"]:
        for example in rec["examples"]:
            assert text[example["source_start"]:example["source_end"]] == example["surface"]
            assert text[example["context_source_start"]:example["context_source_end"]] == example["context"]
    assert replay(make_trace(index, search(index, "赵清岚")))["verified"]


def test_manual_correction_is_optional_and_does_not_seed_name_discovery():
    index = corpus("赵清岚说道：好。\n清岚点头。\n若白。\n欧阳若晨说道：好。\n若晨点头。")
    corrected = rebuild_with_aliases(index, [["赵清岚", "若白"]])
    assert corrected["name_discovery"] == index["name_discovery"]
    assert query_terms(corrected, "赵清岚") == {"赵清岚", "若白"}
    assert query_terms(corrected, "若晨") == {"欧阳若晨", "若晨"}
    assert corrected["aliases"]["suppressed_automatic_groups"]
    cleared = rebuild_with_aliases(corrected, [])
    assert cleared == index
    rebuilt = rebuild_with_book_words(corrected, ["专业术语"])
    assert rebuilt["name_discovery"] == index["name_discovery"]
    assert rebuilt["aliases"] == corrected["aliases"]


def test_discovery_thresholds_are_saved_and_change_index_identity():
    index = corpus("星河说道：好。\n星河点头。")
    config = default_config()
    config["min_contexts"] = 3
    stricter = corpus("星河说道：好。\n星河点头。", person_config=config)
    assert index["index_id"] != stricter["index_id"]
    assert "星河" in index["name_discovery"]["names"]
    assert "星河" not in stricter["name_discovery"]["names"]
    assert replay(make_trace(stricter, search(stricter, "星河")))["verified"]


@pytest.mark.parametrize("field", ["name_discovery", "query_name_decisions", "query_expansions"])
def test_replay_rederives_discovery_and_rejects_modified_rules_or_decisions(field):
    index = corpus("赵清岚说道：好。\n清岚点头。")
    trace = make_trace(index, search(index, "赵清岚"))
    if field == "name_discovery":
        trace["index"][field]["candidates"][0]["accepted"] = False
    else:
        trace["retrieval"][field] = []
    trace["integrity_sha256"] = digest({k: v for k, v in trace.items() if k != "integrity_sha256"})
    with pytest.raises(ValueError):
        replay(trace)


def test_frozen_alias_v12_record_replays_and_upgrade_discards_old_default_pair():
    trace = read_record(Path(__file__).parent / "fixtures" / "alias-v12-trace.json")
    frozen = copy.deepcopy(trace)
    assert trace["engine_version"] == ALIAS_ENGINE_VERSION
    assert replay(trace)["query_policy"] == "book-frequency-aliases/2"
    assert replay(make_trace(trace["index"], trace["retrieval"]))["verified"]
    upgraded = prepare_index_for_search(trace["index"])
    assert upgraded["identity"]["engine_version"] == ENGINE_VERSION
    assert upgraded["aliases"]["manual_groups"] == []
    assert upgraded["aliases"]["automatic_groups"]
    assert trace == frozen


def test_shipped_dictionary_does_not_seed_any_person_pair():
    words = json.loads((Path(__file__).parents[1] / "data" / "dictionary.json").read_text(encoding="utf-8"))
    empty = build_index("普通文字。".encode(), "empty.txt", "utf-8", words)
    assert empty["aliases"]["groups"] == []
    assert empty["name_discovery"]["names"] == []
    assert not {"林黛玉", "黛玉", "贾宝玉", "薛宝钗", "贾母"}.intersection(words)
