"""Sentence boundaries and exact coordinates, independent of tokenization."""

from copy import deepcopy

from evidence.ingest import IngestConfig, ingest
from evidence.sentences import build_sentences


def snapshot(text, max_chars=400):
    raw, cleaned, paragraphs = ingest(text.encode("utf-8"), "utf-8", IngestConfig(max_chars))
    return build_sentences({"source": {"text": raw}, "cleaning": cleaned, "paragraphs": paragraphs})


def texts(text, max_chars=400):
    return [sentence["text"] for sentence in snapshot(text, max_chars)["sentences"]]


def test_sentence_reconstructs_hard_retrieval_cuts_without_crossing_lines():
    text = "甲" * 45 + "。乙；丙，丁。\n戊" * 1
    result = snapshot(text, 16)["sentences"]
    assert [s["text"] for s in result] == ["甲" * 45 + "。", "乙；丙，丁。", "戊"]
    assert result[0]["paragraph_ids"] == ["P000001", "P000002", "P000003"]
    assert result[0]["complete"]
    assert result[-1]["is_fragment"]
    assert result[-1]["fragment_reasons"] == ["no_terminal_punctuation"]


def test_punctuation_runs_semicolons_ellipsis_and_ascii_dot():
    text = "来了？！真的吗!? 分号；逗号，仍是一句。想了想……然后说。Wait... Fine. Done!"
    result = snapshot(text)["sentences"]
    assert [s["text"] for s in result] == [
        "来了？！", "真的吗!?", "分号；逗号，仍是一句。", "想了想……", "然后说。", "Wait...", "Fine.", "Done!",
    ]
    assert result[3]["boundary_rule"] == "SS003"
    assert result[3]["fragment_reasons"] == ["ellipsis"]
    assert result[5]["is_fragment"]
    assert result[6]["boundary_rule"] == "SS005"


def test_dots_do_not_split_decimal_versions_dates_or_abbreviations():
    text = "Dr. Li used v1.2.3 on 2026.10.17 with 3.14 units. Next.\nU.S. office\n例子为3.14，结束。"
    result = snapshot(text)["sentences"]
    assert [s["text"] for s in result] == [
        "Dr. Li used v1.2.3 on 2026.10.17 with 3.14 units.", "Next.", "U.S. office", "例子为3.14，结束。",
    ]
    assert result[0]["complete"]
    assert result[2]["is_fragment"]


def test_closing_quotes_retained_and_speaker_attribution_not_separated():
    result = snapshot("小明说：“我来了。”\n“你去哪儿？”小明问。\n\"Ready?\" Li asked.")["sentences"]
    assert [s["text"] for s in result] == ["小明说：“我来了。”", "“你去哪儿？”小明问。", '\"Ready?\" Li asked.']
    assert result[0]["complete"]
    assert result[1]["fragment_reasons"] == ["quoted_ending_followed_by_text"]
    assert result[2]["is_fragment"]


def test_complex_and_unbalanced_quotes_are_fragments():
    result = snapshot("他说：“先来。再走。”\n他说：“她说‘可以。’。”\n他说：“没有结尾。\n残留闭引号”也保留。")["sentences"]
    assert len(result) == 4
    assert all(s["is_fragment"] for s in result)
    assert "multiple_quoted_sentences" in result[0]["fragment_reasons"]
    assert "nested_quotation" in result[1]["fragment_reasons"]
    assert "unclosed_quotation" in result[2]["fragment_reasons"]
    assert "unmatched_closing_quote" in result[3]["fragment_reasons"]


def test_straight_single_quotes_and_apostrophes_are_distinguished():
    result = snapshot("'Ready?' Li asked.\nDon’t split students' words.\n她说：‘don’t。’")["sentences"]
    assert [s["text"] for s in result] == ["'Ready?' Li asked.", "Don’t split students' words.", "她说：‘don’t。’"]
    assert result[0]["is_fragment"]
    assert result[1]["complete"]
    assert result[2]["complete"]


def test_headings_excluded_no_join_across_lines_or_chapters():
    result = snapshot("第一章 开始\n上一行未完\n下一行。\n第二章 继续\n另一句。", 16)["sentences"]
    assert [s["text"] for s in result] == ["上一行未完", "下一行。", "另一句。"]
    assert [s["chapter"] for s in result] == ["第一章 开始", "第一章 开始", "第二章 继续"]
    assert [s["id"] for s in result] == ["S000001", "S000002", "S000003"]


def test_raw_slices_and_coordinates_with_bom_crlf_tabs_fullwidth_and_emoji():
    raw = "\ufeff　第一章 文章\r\n\r\n  甲\t😀来了。　乙走了！　\r末行\n"
    result = snapshot(raw)["sentences"]
    assert [s["text"] for s in result] == ["甲 😀来了。", "乙走了！", "末行"]
    for sentence in result:
        assert raw[sentence["source_start"]:sentence["source_end"]] == sentence["source_text"]
    assert result[0]["source_text"] == "甲\t😀来了。"
    assert result[0]["location"] == {"start_line": 3, "start_column": 3, "end_line": 3, "end_column": 8}
    assert result[1]["location"]["start_column"] == 10
    assert result[2]["location"]["start_line"] == 4


def test_deterministic_snapshot_isolated_from_returned_rule_mutations():
    first = snapshot("文章定义是明确的。\n还有一个片段")
    expected = deepcopy(first)
    assert first == snapshot("文章定义是明确的。\n还有一个片段")
    first["config"]["exclude_headings"] = False
    first["rules"]["SS001"] = "被修改"
    assert snapshot("文章定义是明确的。\n还有一个片段") == expected
