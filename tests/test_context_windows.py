"""Window statistics, real source spans, and conservative join boundaries."""

from copy import deepcopy
import math

import pytest

from evidence.context import (CONTEXT_CONFIG, boundary_between, build_context_index, search_windows)
from evidence.ingest import IngestConfig, SourceMap, ingest
from evidence.sentences import build_sentences
from evidence.tokenization import Tokenizer


def corpus(text, max_chars=400):
    raw, cleaned, paragraphs = ingest(text.encode("utf-8"), "utf-8", IngestConfig(max_chars))
    mapping, tokenizer = SourceMap(cleaned["source_runs"], raw), Tokenizer([])
    for paragraph in paragraphs:
        paragraph["tokens"], _ = tokenizer.tokenize(paragraph["text"], paragraph["clean_start"], mapping)
    index = {"index_id": "fixture", "source": {"text": raw}, "cleaning": cleaned,
             "paragraphs": paragraphs, "query_expansion": {"book_matches": {}}}
    index["sentences"] = build_sentences(index)
    index["context"] = build_context_index(index)
    return index


def query(*features):
    return {"config": {"k1": 1.5, "b": .75, "top_k": 5},
            "query_terms": [{"term": feature, "kind": "token", "weight": 1, "df": 999, "idf": 999}
                            if isinstance(feature, str) else feature for feature in features]}


def unit(index, *sentence_ids):
    return next(unit for unit in index["context"]["units"] if unit["sentence_ids"] == list(sentence_ids))


def test_window_corpus_has_independent_overlap_safe_hand_calculated_bm25():
    index = corpus("甲甲乙。\n甲丙。\n乙。")
    stats = index["context"]["statistics"]
    assert stats["N"] == stats["unit_count"] == 6
    assert stats["total_tokens"] == 20
    assert stats["avgdl"] == 20 / 6
    assert stats["document_frequency"] == {"丙": 4, "乙": 5, "甲": 5}
    assert stats["collection_tf"] == {"丙": 4, "乙": 6, "甲": 10}
    ranked = search_windows(index, query("甲", "丁"))
    assert ranked["candidate_count"] == 5
    assert ranked["query_terms"][0]["df"] == 5
    expected_idf = math.log1p(1.5 / 5.5)
    assert ranked["query_terms"][0]["idf"] == pytest.approx(expected_idf)
    first = next(c for c in ranked["candidates"] if c["sentence_ids"] == ["S000001"])
    norm = .25 + .75 * 3 / (20 / 6)
    assert first["score"] == pytest.approx(expected_idf * 5 / (2 + 1.5 * norm))
    assert first["dl"] == 3
    assert first["term_details"][0]["tf"] == 2
    assert len(first["term_details"][0]["occurrences"]) == 2
    assert first["term_details"][1]["tf"] == 0
    assert first["term_details"][1]["contribution"] == 0
    assert first["term_details"][1]["idf"] == pytest.approx(math.log(14))
    assert all(c["score"] == sum(t["contribution"] for t in c["term_details"]) for c in ranked["candidates"])
    assert ranked["candidates"] == sorted(ranked["candidates"], key=lambda c: (-c["score"], c["source_start"], c["window_id"]))


def test_newlines_blank_lines_cleaning_and_source_spans_are_preserved():
    raw = "\ufeff　甲\t😀启动。　\r\n\r\n  该项目运行。\r下一句。"
    index = corpus(raw)
    combined = unit(index, "S000001", "S000002")
    assert combined["text"] == "甲 😀启动。\n\n该项目运行。"
    assert combined["source_text"] == "甲\t😀启动。　\r\n\r\n  该项目运行。"
    assert combined["paragraph_ids"] == ["P000001", "P000002"]
    assert len(combined["sentence_spans"]) == 2
    assert combined["location"] == {"start_line": 1, "start_column": 3, "end_line": 3, "end_column": 8}
    for candidate in index["context"]["units"]:
        assert raw[candidate["source_start"]:candidate["source_end"]] == candidate["source_text"]
        assert index["cleaning"]["text"][candidate["clean_start"]:candidate["clean_end"]] == candidate["text"]
        for span in candidate["sentence_spans"]:
            assert raw[span["source_start"]:span["source_end"]] == span["source_text"]
    assert len(index["context"]["boundaries"]) == 0


def test_chapters_and_subheadings_stop_crossing_without_discarding_fragments():
    index = corpus("第一章 项目\n甲项目启动。\n项目概况\n该项目运行。\n第二章 继续\n乙项目启动。")
    sentences = index["sentences"]["sentences"]
    assert [s["text"] for s in sentences] == ["甲项目启动。", "项目概况", "该项目运行。", "乙项目启动。"]
    assert all(u["sentence_count"] == 1 for u in index["context"]["units"])
    assert "CW003" in boundary_between(index, sentences[0], sentences[1])
    assert "CW003" in boundary_between(index, "S000003", "S000004")
    assert unit(index, "S000002")["text"] == "项目概况"


def test_separate_list_items_and_markdown_heading_do_not_merge():
    index = corpus("（一）甲项目启动。后续运行。\n（二）乙项目启动。后续运行。\n# 小结\n普通文字。")
    first = unit(index, "S000001", "S000002")
    second = unit(index, "S000003", "S000004")
    assert first["text"] == "（一）甲项目启动。后续运行。"
    assert second["text"] == "（二）乙项目启动。后续运行。"
    assert "CW004" in boundary_between(index, "S000002", "S000003")
    assert "CW004" in boundary_between(index, "S000004", "S000005")
    assert not any("S000002" in u["sentence_ids"] and "S000003" in u["sentence_ids"] for u in index["context"]["units"])


@pytest.mark.parametrize("prefix", ["甲说：“项目启动。”", "“项目启动。", "项目未完", "项目想了想……"])
def test_quotations_and_fragments_remain_singletons(prefix):
    index = corpus(prefix + "\n该项目运行。\n还有后续。")
    assert "CW005" in boundary_between(index, "S000001", "S000002")
    assert all(u["sentence_count"] == 1 for u in index["context"]["units"] if "S000001" in u["sentence_ids"])
    assert unit(index, "S000002", "S000003")["sentence_count"] == 2


def test_windows_have_at_most_four_sentences_and_actual_character_cap():
    index = corpus("甲。\n乙。\n丙。\n丁。\n戊。")
    units = index["context"]["units"]
    assert max(u["sentence_count"] for u in units) == 4
    assert len(units) == 14  # 5 singles + 4 pairs + 3 triples + 2 quadruples.
    index["context"] = build_context_index(index, {**CONTEXT_CONFIG, "max_chars": 16})
    assert all(len(u["text"]) <= 16 for u in index["context"]["units"])
    capped = corpus("甲" * 7 + "。\n" + "乙" * 7 + "。")
    capped["context"] = build_context_index(capped, {**CONTEXT_CONFIG, "max_chars": 16})
    # Each sentence has 8 characters; their real intervening newline makes
    # a two-sentence window 17 characters, so even this pair must be omitted.
    assert all(u["sentence_count"] == 1 for u in capped["context"]["units"])
    long = corpus("甲" * 650 + "。\n乙。", max_chars=16)
    assert len(unit(long, "S000001")["paragraph_ids"]) > 1
    assert unit(long, "S000001")["length"] == 650
    assert len(long["context"]["units"]) == 2
    assert len(unit(long, "S000001")["text"]) == 651


def test_hard_paragraph_cuts_do_not_duplicate_original_token_occurrences():
    text = "甲" * 32 + "乙。丙。"
    full, split = corpus(text), corpus(text, max_chars=16)
    assert full["context"]["statistics"] == split["context"]["statistics"]
    combined = unit(split, "S000001", "S000002")
    assert len(combined["tokens"]) == 34
    assert combined["term_frequencies"]["甲"] == 32
    original = split["paragraphs"][0]["tokens"][0]
    split["paragraphs"][0]["tokens"].append(deepcopy(original))
    repeated = build_context_index(split)
    assert repeated["statistics"] == full["context"]["statistics"]


def test_window_tokens_keep_only_lightweight_source_fields():
    index = corpus("甲乙。\n丙丁。\n戊。")
    original = index["paragraphs"][0]["tokens"][0]
    original["dag_metadata"] = {"dictionary_entries": ["large frozen explanation"] * 50}
    before = deepcopy(index["paragraphs"])
    snapshot = build_context_index(index)
    allowed = {"term", "surface", "clean_start", "clean_end", "source_start", "source_end"}
    assert all(set(token) == allowed for u in snapshot["units"] for token in u["tokens"])
    assert index["paragraphs"] == before
    assert index["paragraphs"][0]["tokens"][0]["dag_metadata"]["dictionary_entries"]


def test_expression_matches_reuse_only_original_contiguous_phrase_spans():
    index = corpus("数据收集。\n收集数据。\n甲数。 据乙。")
    # Token offsets refer to the original frozen paragraph tokens.
    index["query_expansion"]["book_matches"] = {
        "数据": {"P000001": [[0, 2]], "P000002": [[2, 4]]},
        # These two tokens have punctuation and whitespace between them.
        "数据伪造": {"P000003": [[1, 3]]},
    }
    index["context"] = build_context_index(index)
    matches = index["context"]["expression_matches"]
    assert "数据伪造" not in matches
    combined = unit(index, "S000001", "S000002")
    positions = matches["数据"][combined["id"]]
    assert len(positions) == 2
    for occurrence in positions:
        assert index["source"]["text"][occurrence["source_start"]:occurrence["source_end"]] == "数据"
    ranked = search_windows(index, query({"term": "数据", "kind": "expression", "weight": .7}))
    candidate = next(c for c in ranked["candidates"] if c["window_id"] == combined["id"])
    assert candidate["dl"] == 8
    assert candidate["term_details"][0]["tf"] == 2
    assert ranked["query_terms"][0]["df"] == len(matches["数据"])


def test_phrase_is_not_inferred_across_lines_even_when_window_contains_both_words():
    index = corpus("甲数。\n据乙。")
    index["query_expansion"]["book_matches"] = {"数据": {}}
    index["context"] = build_context_index(index)
    assert unit(index, "S000001", "S000002")["sentence_count"] == 2
    assert search_windows(index, query({"term": "数据", "kind": "expression", "weight": 1}))["candidates"] == []


def test_group_maximum_covered_features_and_feature_order_ties():
    index = corpus("甲乙。\n丙。")
    result = search_windows(index, query(
        {"term": "甲", "kind": "token", "weight": 1, "score_group": "g"},
        {"term": "乙", "kind": "token", "weight": 1, "score_group": "g"},
        {"term": "丙", "kind": "token", "weight": 1, "covered_by_expression": True},
    ))
    combined = next(c for c in result["candidates"] if c["sentence_count"] == 2)
    first, second, covered = combined["term_details"]
    assert first["raw_contribution"] == second["raw_contribution"]
    assert first["score_reason"] == "group_maximum"
    assert second["score_reason"] == "same_group_not_selected"
    assert covered["score_reason"] == "covered_by_expression"
    assert second["contribution"] == covered["contribution"] == 0
    assert combined["score"] == first["raw_contribution"]
    assert len(result["candidates"]) == 2  # Covered-only singleton never recalls.


def test_empty_token_units_zero_average_and_no_mutation_are_safe():
    index = corpus("！！\n？")
    assert index["context"]["statistics"]["N"] == 0
    assert index["context"]["statistics"]["avgdl"] == 0
    assert search_windows(index, query("甲"))["candidates"] == []
    index = corpus("甲。\n乙。")
    before = deepcopy(index)
    snapshot = build_context_index(index)
    search_windows(index, query("甲"))
    assert index == before
    snapshot["config"]["max_chars"] = 100
    snapshot["rules"]["CW001"] = "modified"
    assert build_context_index(index) == index["context"]


@pytest.mark.parametrize("changes", [{"max_sentences": 1}, {"max_chars": 0}, {"max_chars": True},
                                      {"min_window_sentences": 3}, {"subheading_max_chars": 500}])
def test_invalid_context_configuration_is_rejected(changes):
    with pytest.raises(ValueError, match="窗口配置"):
        build_context_index(corpus("甲。"), {**CONTEXT_CONFIG, **changes})
