"""Lexical recall, real phrase evidence, bounded expansion and frozen replay."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from evidence import DAG_ENGINE_VERSION, ENGINE_VERSION
from evidence.dag_tokenization import lexicon_snapshot
from evidence.query_expansion import DEFAULT_CONFIG, default_resource, resource_snapshot
from evidence.retrieval import (build_index, configure_query_expansion, digest,
                                rebuild_with_aliases, rebuild_with_book_words, search)
from evidence.tracing import make_trace, read_record, replay, serialize

ROOT = Path(__file__).resolve().parents[1]


def resource(*groups):
    return resource_snapshot([{"id": f"sense:{i}", "pos": "v", "forms": list(forms)}
                              for i, forms in enumerate(groups)], {"name": "controlled-test"})


def corpus(text, **kwargs):
    kwargs.setdefault("lexicon", lexicon_snapshot({}))
    return build_index(text.encode(), "表达回归.txt", "utf-8", kwargs.pop("words", []), **kwargs)


def by_term(result):
    return {t["term"]: t for t in result["query_terms"]}


def resign(trace):
    trace["integrity_sha256"] = digest({k: v for k, v in trace.items() if k != "integrity_sha256"})


def test_bundled_wordnet_is_general_and_keeps_senses_and_project_corrections_separate():
    data = default_resource()
    assert len(data["groups"]) > 10000
    assert len({t for g in data["groups"] for t in g["forms"]}) > 25000
    assert data["source"]["commit"] == "19256777c557e400dd5ef592428fa78def6c07dc"
    assert any({"电脑", "计算机"}.issubset(g["forms"]) and g["origin"] == "wordnet" for g in data["groups"])
    assert any({"赠送", "送给"}.issubset(g["forms"]) and g["origin"] == "wordnet" for g in data["groups"])
    assert any("赠与" in g["forms"] and g["origin"] == "reviewed" for g in data["groups"])


def test_gift_query_recalls_variants_without_matching_send_off_or_hospital():
    index = corpus("甲把书赠与乙。\n乙把书送给甲。\n甲送别乙。\n乙送医。", words=["赠送"])
    frozen = copy.deepcopy(index)
    result = search(index, "赠送")
    assert set(result["top_ids"]) == {"P000001", "P000002"}
    assert {e["term"] for e in result["lexical_expansions"]} == {"赠与", "送给"}
    assert {e["origin"] for e in result["lexical_expansions"]} == {"reviewed"}
    assert "赠与" not in index["statistics"]["collection_tf"]  # split into characters
    for candidate in result["candidates"]:
        for detail in candidate["term_details"]:
            for span in detail["occurrences"]:
                assert index["source"]["text"][span["source_start"]:span["source_end"]] == detail["term"]
    assert index == frozen
    disabled = configure_query_expansion(index, config={**DEFAULT_CONFIG, "enabled": False})
    assert search(disabled, "赠送")["status"] == "no_matches"


def test_real_wordnet_recalls_computer_without_manual_word_pair():
    index = corpus("他在使用电脑。\n桌上放着一本书。")
    result = search(index, "计算机")
    assert result["top_ids"] == ["P000001"]
    assert result["lexical_expansions"][0]["term"] == "电脑"
    assert result["lexical_expansions"][0]["origin"] == "wordnet"
    assert "03082979-n" in result["lexical_expansions"][0]["sense_ids"]


def test_expression_matches_require_token_boundaries_and_never_cross_gaps():
    index = corpus("送给。\n送，给。\n送 给。\n送礼给。\n送给甲。",
                   words=["送给甲"], expansion_resource=resource(("赠送", "送给")))
    result = search(index, "赠送")
    assert result["top_ids"] == ["P000001"]
    assert list(index["query_expansion"]["book_matches"]["送给"]) == ["P000001"]


def test_split_query_phrase_retains_offsets_and_suppresses_only_covered_fragments():
    index = corpus("赠送。\n送医。", expansion_resource=resource(("赠送", "送给")))
    result = search(index, "\ufeff　送给？")
    terms = by_term(result)
    assert [t["term"] for t in result["query_tokens"]] == ["送", "给"]
    assert terms["送给"]["kind"] == "expression"
    assert terms["送给"]["query_frequency"] == 1
    assert terms["送"]["covered_by_expression"]
    assert result["top_ids"] == ["P000001"]
    for e in result["lexical_expansions"]:
        assert result["query"][e["query_source_start"]:e["query_source_end"]] == "送给"
    independent = search(index, "送给 送")
    assert not by_term(independent)["送"].get("covered_by_expression")
    assert set(independent["top_ids"]) == {"P000001", "P000002"}


@pytest.mark.parametrize("query", ["赠送 送给 送给", "送给 送给 赠送"])
def test_expanded_phrase_that_also_occurs_in_query_keeps_literal_weight_and_positions(query):
    index = corpus("送给。", expansion_resource=resource(("赠送", "送给")))
    result = search(index, query)
    feature = by_term(result)["送给"]
    assert feature["weight"] == 1
    assert feature["query_frequency"] == 2
    literal_sources = [s for s in feature["sources"] if s["kind"] == "literal_expression"]
    assert len(literal_sources) == 2
    assert all(result["query"][s["query_source_start"]:s["query_source_end"]] == "送给" for s in literal_sources)


def test_missing_alternatives_are_logged_and_expansion_does_not_recurse():
    index = corpus("赠与。\n送给。", expansion_resource=resource(("赠送", "赠与", "馈赠"), ("赠与", "送给")))
    result = search(index, "赠送")
    assert [e["term"] for e in result["lexical_expansions"]] == ["赠与"]
    assert result["top_ids"] == ["P000001"]
    rejected = {c["term"]: c["reason"] for c in result["lexical_decisions"][0]["candidates"]}
    assert rejected["馈赠"] == "absent_from_book"
    assert "送给" not in by_term(result)


def test_group_maximum_deduplicates_sources_and_preserves_sum_of_actual_contributions():
    index = corpus("赠送赠与送给。\n赠与。", words=["赠送", "赠与", "送给"],
                   expansion_resource=resource(("赠送", "赠与", "送给")))
    result = search(index, "赠送 赠送 送给")
    assert len(result["query_terms"]) == 3
    assert by_term(result)["赠送"]["query_frequency"] == 2
    assert by_term(result)["送给"]["weight"] == 1
    assert by_term(result)["赠与"]["weight"] == 0.7
    for candidate in result["candidates"]:
        details = candidate["term_details"]
        assert candidate["score"] == sum(t["contribution"] for t in details)
        assert candidate["score"] == max(t["raw_contribution"] for t in details)
        assert sum(t["contribution"] > 0 for t in details) == 1


def test_expansion_caps_and_ambiguities_have_explicit_decisions():
    index = corpus("赠与。\n送给。\n馈赠。\n逝世。", words=["赠送", "去世"],
        expansion_resource=resource(("赠送", "赠与", "送给"), ("赠送", "馈赠"), ("去世", "逝世")),
        expansion_config={**DEFAULT_CONFIG, "max_alternatives": 1, "max_sources": 1})
    result = search(index, "赠送 去世")
    assert len(result["lexical_expansions"]) == 1
    assert result["lexical_decisions"][0]["ambiguous"]
    reasons = {c["reason"] for d in result["lexical_decisions"] for c in d["candidates"]}
    assert {"alternative_limit", "source_limit"}.issubset(reasons)


def test_names_status_words_and_blocked_expressions_are_not_expanded():
    index = corpus("赵清岚说道：没有赠与。\n清岚点头。\n赠送。", words=["没有", "赠送", "赠与"],
        alias_groups=[["赵清岚", "清岚"]], blocked_expressions=["赠与"],
        expansion_resource=resource(("赵清岚", "赠送"), ("没有", "赠送"), ("赠送", "赠与")))
    result = search(index, "赵清岚 没有 赠送")
    assert result["query_expansions"]  # person aliases still work
    assert result["lexical_expansions"] == []
    assert by_term(result)["没有"]["weight"] == 1
    assert any(c["reason"] == "excluded_expression" for d in result["lexical_decisions"] for c in d["candidates"])


def test_book_corrections_override_general_groups_without_changing_main_statistics():
    index = corpus("赠与。\n送给。\n馈赠。", expansion_resource=resource(("赠送", "赠与", "送给")))
    changed = configure_query_expansion(index, groups=[["赠送", "馈赠"]])
    assert index["statistics"] == changed["statistics"]
    assert index["paragraphs"] == changed["paragraphs"]
    assert index["postings"] == changed["postings"]
    assert index["index_id"] != changed["index_id"]
    result = search(changed, "赠送")
    assert [e["term"] for e in result["lexical_expansions"]] == ["馈赠"]
    assert result["lexical_expansions"][0]["origin"] == "manual"
    for rebuilt in [rebuild_with_book_words(changed, ["赠送"]), rebuild_with_aliases(changed, [])]:
        assert rebuilt["query_expansion"]["manual_groups"] == changed["query_expansion"]["manual_groups"]
        assert rebuilt["query_expansion"]["resource"] == changed["query_expansion"]["resource"]
    assert configure_query_expansion(changed, groups=[])["index_id"] == index["index_id"]


@pytest.mark.parametrize("config", [{"weight": 0}, {"weight": float("nan")}, {"weight": True},
                                     {"enabled": 1}, {"max_alternatives": 0}, {"max_sources": 31}])
def test_invalid_settings_rejected(config):
    with pytest.raises(ValueError):
        corpus("赠与。", expansion_config={**DEFAULT_CONFIG, **config}, expansion_resource=resource(("赠送", "赠与")))


@pytest.mark.parametrize("groups", [[["赠送"]], [["送", "赠送"]], [["没有", "赠送"]],
                                     [["赠送", "赠与"], ["赠与", "送给"]]])
def test_invalid_manual_groups_rejected(groups):
    with pytest.raises(ValueError):
        corpus("赠与。", expression_groups=groups)


def test_trace_contains_frozen_resources_and_replays_without_any_installed_resource(tmp_path, monkeypatch):
    import evidence.query_expansion as expansion
    index = corpus("电脑。")
    result = search(index, "计算机")
    trace = make_trace(index, result)
    path = tmp_path / "lexical.json.gz"
    path.write_bytes(serialize(trace))
    monkeypatch.setattr(expansion, "default_resource", lambda: pytest.fail("Replay must use the embedded resource"))
    assert replay(trace)["verified"]
    code = "from evidence.query_expansion import default_resource; import evidence.query_expansion as e; e.default_resource=lambda: (_ for _ in ()).throw(RuntimeError('offline resource forbidden')); from evidence.tracing import read_record,replay; import sys; print(replay(read_record(sys.argv[1]))['verified'])"
    process = subprocess.run([sys.executable, "-c", code, str(path)], cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "True"


@pytest.mark.parametrize("target", ["expansion", "phrase", "contribution", "resource"])
def test_resigned_tampering_still_fails_recomputation(target):
    index = corpus("赠与。", expansion_resource=resource(("赠送", "赠与")))
    trace = make_trace(index, search(index, "赠送"))
    if target == "expansion":
        trace["retrieval"]["lexical_expansions"][0]["weight"] = 1
    elif target == "phrase":
        trace["index"]["query_expansion"]["book_matches"]["赠与"]["P000001"][0][0] += 1
    elif target == "contribution":
        trace["retrieval"]["candidates"][0]["term_details"][-1]["contribution"] += 1
    else:
        trace["index"]["query_expansion"]["resource"]["groups"][0]["forms"].append("送给")
    resign(trace)
    with pytest.raises(ValueError):
        replay(trace)


def test_frozen_v14_trace_keeps_original_query_policy_and_reexports_unchanged():
    old = read_record(ROOT / "tests/fixtures/dag-v14-trace.json")
    assert old["engine_version"] == DAG_ENGINE_VERSION
    assert replay(old)["verified"]
    assert make_trace(old["index"], old["retrieval"])["engine_version"] == DAG_ENGINE_VERSION
    assert "lexical_expansions" not in old["retrieval"]
    current = search(old["index"], "赠送")
    assert len(current["top_ids"]) == 2
    trace = make_trace(old["index"], current)
    assert trace["engine_version"] == ENGINE_VERSION
    assert replay(trace)["verified"]


def test_cli_can_disable_expansion_and_export_a_replayable_trace(tmp_path):
    index = corpus("赠与。", words=["赠送"])
    source, trace_path = tmp_path / "index.json.gz", tmp_path / "trace.json.gz"
    source.write_bytes(serialize(index))
    process = subprocess.run([sys.executable, "-m", "evidence.cli", "search", str(source), "赠送", "--no-expansion", "--export", str(trace_path)],
                             cwd=ROOT, capture_output=True, encoding="utf-8")
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["status"] == "no_matches"
    assert replay(read_record(trace_path))["verified"]
