"""End-to-end acceptance examples using article excerpts and original prose."""

import json

import pytest

from evidence.evaluate import ARTICLE_DIR, evaluate_articles, load_articles


CASES = json.loads((ARTICLE_DIR / "cases.json").read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="module")
def article_report():
    return evaluate_articles()


def test_article_fixtures_have_attribution_and_unchanged_bytes():
    catalog, cases, payloads = load_articles()
    assert len(cases) == 26
    assert len(payloads) == 16
    sourced = [source for source in catalog["sources"] if source["kind"] == "sourced_excerpt"]
    assert len(sourced) == 3
    assert all(source["url"].startswith("https://") and source["published"] and source["fetched"] == "2026-10-17"
               for source in sourced)
    # The source title/date metadata must not become implicit answer evidence.
    assert payloads["change6-launch.txt"].decode("utf-8").strip() == (
        "嫦娥六号探测器于5月3日在中国文昌航天发射场发射升空并进入地月转移轨道。")
    assert "2024" not in payloads["change6-launch.txt"].decode("utf-8")
    context_sources = [source for source in catalog["sources"] if source["fixture"].startswith("context-")]
    assert len(context_sources) == 9
    assert all(source["kind"] == "original_test_article" and "不是网页摘录" in source["note"]
               for source in context_sources)
    # These fixtures contain the same sentences, with only their physical
    # layout changed. Neither version pretends to repair or join a sentence.
    assert payloads["context-project-newlines.txt"].decode("utf-8").replace("\n", "") == (
        payloads["context-project-sameline.txt"].decode("utf-8").replace("\n", ""))


@pytest.mark.parametrize("case_id", [case["id"] for case in CASES])
def test_article_question_contract(article_report, case_id):
    record = next(row for row in article_report["cases"] if row["id"] == case_id)
    assert record["passed"], json.dumps(record, ensure_ascii=False, indent=2)


def test_context_article_report_retains_window_statistics_and_joint_roles(article_report):
    rows = {row["id"]: row for row in article_report["cases"]}
    row = rows["context-time-newlines"]
    assert row["window_statistics"]["N"] > row["window_statistics"]["singleton_count"]
    assert row["window_statistics"]["avgdl"] > 0
    assert row["answer"]["bindings"][0]["antecedent"]["text"] == "青禾社区雨水收集项目"
    group = next(group for group in row["evidence_groups"] if group["sentence_ids"] == ["S000001", "S000002"])
    assert group["supporting_sentence_ids"] == ["S000001"]
    assert group["direct_sentence_ids"] == ["S000002"]
    assert [span["role"] for span in group["source_spans"]] == ["support", "direct"]
    assert [span["source_text"] for span in group["source_spans"]] == row["expected_evidence_sentences"]
    assert rows["context-time-sameline"]["answer"]["fields"]["time"]["text"] == row["answer"]["fields"]["time"]["text"]
    conflict = rows["context-conflicting-pairs"]
    assert len(conflict["answer"]["evidence_group_ids"]) == 2
    assert all(group["binding"] for group in conflict["evidence_groups"] if group["is_answer_support"])
    overlap = rows["context-overlap-proof-dedup"]
    assert overlap["checks"]["groups_deduplicated"]
    assert overlap["checks"]["overlapping_windows_share_proof"]
