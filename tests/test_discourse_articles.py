"""Finite article acceptance tests for deterministic causal/method answers."""

import hashlib
import json

import pytest

from evidence import evaluate_discourse


CASES = json.loads((evaluate_discourse.ARTICLE_DIR / "cases.json").read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="module")
def discourse_report():
    return evaluate_discourse.evaluate_articles()


def test_discourse_fixture_attribution_and_original_article_bodies():
    catalog, cases, payloads = evaluate_discourse.load_articles()
    assert len(cases) == 36
    assert len(payloads) == 31
    originals = [source for source in catalog["sources"] if source["kind"] == "original_test_article"]
    sourced = [source for source in catalog["sources"] if source["kind"] == "sourced_excerpt"]
    assert len(originals) == 28
    assert len(sourced) == 3
    assert all("原创测试短文" in source["title"] and "不是网页摘录" in source["note"] for source in originals)
    assert all(payloads[source["fixture"]].decode("utf-8").count("。") >= 3 for source in originals)
    for source in sourced:
        original_path = evaluate_discourse.ARTICLE_DIR.parents[1] / source["copied_from"]
        assert payloads[source["fixture"]] == original_path.read_bytes()
        assert source["url"].startswith("https://") and source["published"] and source["fetched"] == "2026-10-17"
    assert payloads["adjacent-reason-newlines.txt"].decode("utf-8").replace("\n", "") == payloads["adjacent-reason-sameline.txt"].decode("utf-8").replace("\n", "")


@pytest.mark.parametrize("case_id", [case["id"] for case in CASES])
def test_discourse_article_contract(discourse_report, case_id):
    record = next(row for row in discourse_report["cases"] if row["id"] == case_id)
    assert record["passed"], json.dumps(record, ensure_ascii=False, indent=2)


def test_adjacent_explanation_preserves_two_real_spans_and_roles(discourse_report):
    rows = {row["id"]: row for row in discourse_report["cases"]}
    newline, sameline = rows["adjacent-reason-newlines"], rows["adjacent-reason-sameline"]
    assert newline["answer"]["fields"]["reason"]["text"] == sameline["answer"]["fields"]["reason"]["text"] == "入口处的水管破裂"
    assert newline["answer"]["bindings"]
    assert newline["answer"]["fields"]["reason"]["sentence_id"] == "S000002"
    group = next(group for group in newline["evidence_groups"] if group["id"] in newline["answer"]["evidence_group_ids"])
    assert group["sentence_ids"] == ["S000001", "S000002"]
    assert [span["role"] for span in group["source_spans"]] == ["support", "direct"]
    assert [span["source_text"] for span in group["source_spans"]] == newline["expected_evidence_sentences"]


def test_conflicts_keep_both_proof_groups_without_factual_fields(discourse_report):
    rows = {row["id"]: row for row in discourse_report["cases"]}
    for case_id in ("reason-conflict", "method-conflict"):
        row = rows[case_id]
        assert row["answer"]["status"] == "ambiguous"
        assert not row["answer"]["fields"]
        assert len(row["answer"]["evidence_group_ids"]) == 2
        assert sum(decision["accepted"] for decision in row["answer"]["decisions"]) == 2


def test_catalog_hash_mismatch_is_rejected(tmp_path):
    payload = b"fixture\n"
    (tmp_path / "fixture.txt").write_bytes(payload + b"changed\n")
    (tmp_path / "sources.json").write_text(json.dumps({"sources": [{"fixture": "fixture.txt", "text_sha256": hashlib.sha256(payload).hexdigest(), "byte_length": len(payload)}]}), encoding="utf-8")
    (tmp_path / "cases.json").write_text('{"cases": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="快照字节"):
        evaluate_discourse.load_articles(tmp_path)


def test_cli_writes_replayable_json(discourse_report, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(evaluate_discourse, "evaluate_articles", lambda folder: discourse_report)
    destination = tmp_path / "report.json"
    code = evaluate_discourse.main(["--output", str(destination)])
    assert code == (0 if discourse_report["summary"]["all_passed"] else 1)
    restored = json.loads(destination.read_text(encoding="utf-8"))
    assert restored["summary"] == discourse_report["summary"]
    assert restored["cases"][0]["resources"]["source_bytes_sha256"]
    assert "reason-because" in capsys.readouterr().out
