"""Run the small, attributed article QA acceptance set.

This is a finite regression check, not a general estimate of QA accuracy.
Run with ``python -m evidence.evaluate --output .data/article-evaluation.json``.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .retrieval import build_index, search


ARTICLE_DIR = Path(__file__).resolve().parents[1] / "data" / "articles"


def load_articles(folder=ARTICLE_DIR):
    folder = Path(folder)
    catalog = json.loads((folder / "sources.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
    sources = {source["fixture"]: source for source in catalog["sources"]}
    payloads = {}
    for name, source in sources.items():
        if Path(name).name != name:
            raise ValueError("文章文件名不得含目录。")
        payload = (folder / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != source["text_sha256"]:
            raise ValueError(f"文章快照哈希不一致：{name}")
        if len(payload) != source["byte_length"]:
            raise ValueError(f"文章快照长度不一致：{name}")
        payloads[name] = payload
    for case in manifest["cases"]:
        if case["fixture"] not in payloads:
            raise ValueError(f"测试使用了未注明来源的文章：{case['fixture']}")
    return catalog, manifest["cases"], payloads


def evaluate_articles(folder=ARTICLE_DIR):
    """Return explicit per-case observations and checks without network use."""
    catalog, cases, payloads = load_articles(folder)
    indexes = {}
    records = []
    for case in cases:
        name = case["fixture"]
        if name not in indexes:
            indexes[name] = build_index(payloads[name], name, "utf-8", [])
        index = indexes[name]
        result = search(index, case["query"])
        answer = result["answer"]
        evidence = result["sentence_evidence"]["candidates"]
        expected = case["expected_status"]
        checks = {"answer_state": answer["status"] != "answered" if expected == "not_answered"
                  else answer["status"] == expected}
        if expected != "answered":
            checks["no_unsupported_fields"] = not answer["fields"]
        if case.get("expected_field"):
            field = case["expected_field"]
            checks["field_value"] = answer["fields"].get(field["name"], {}).get("text") == field["text"]
        if case.get("expected_evidence"):
            checks["evidence_found"] = any(item["source_text"] == case["expected_evidence"] for item in evidence)
        if case.get("expected_evidence_sentences"):
            checks["evidence_found"] = all(
                any(item["source_text"] == text for item in evidence)
                for text in case["expected_evidence_sentences"])
        bindings = answer.get("bindings", [])
        if "expected_binding" in case:
            checks["binding_expected"] = bool(bindings) == case["expected_binding"]
        groups = result["evidence_groups"]["groups"]
        if case.get("expected_group_sentence_ids"):
            wanted = case["expected_group_sentence_ids"]
            matching_groups = [group for group in groups if group["sentence_ids"] == wanted]
            checks["expected_group_found"] = len(matching_groups) == 1
            if case.get("expected_min_proof_windows"):
                checks["overlapping_windows_share_proof"] = bool(matching_groups) and len(
                    matching_groups[0]["window_ids"]) >= case["expected_min_proof_windows"]
        checks["groups_deduplicated"] = len(groups) == len({tuple(group["sentence_ids"]) for group in groups})
        checks["field_source_spans"] = all(
            index["source"]["text"][field["source_start"]:field["source_end"]] == field["text"]
            for field in answer["fields"].values()
        )
        proof_bindings = [*bindings, *(group["binding"] for group in groups if group.get("binding"))]
        checks["binding_source_spans"] = all(
            index["source"]["text"][binding[name]["source_start"]:binding[name]["source_end"]] == binding[name]["text"]
            for binding in proof_bindings for name in ("antecedent", "reference"))
        by_id = {item["sentence_id"]: item for item in evidence}
        checks["group_source_spans"] = all(
            span["sentence_id"] in by_id
            and index["source"]["text"][span["source_start"]:span["source_end"]] == by_id[span["sentence_id"]]["source_text"]
            for group in groups for span in group["source_spans"])
        if expected == "answered":
            checks["answer_has_evidence"] = bool(answer["evidence_ids"])
        question = result["question"]
        # No example here asks for these expressions literally; they must guide
        # intent detection, rather than enter the ordinary retrieval vocabulary.
        checks["question_words_filtered"] = not set(question["keywords"]).intersection(
            {"为什么", "为何", "怎样", "如何", "什么时候", "多少", "是什么", "谁"})
        records.append({
            "id": case["id"], "fixture": name, "query": case["query"],
            "purpose": case["purpose"], "expected_status": expected,
            "expected_field": case.get("expected_field"),
            "expected_evidence": case.get("expected_evidence"),
            "expected_evidence_sentences": case.get("expected_evidence_sentences"),
            "expected_binding": case.get("expected_binding"),
            "expected_group_sentence_ids": case.get("expected_group_sentence_ids"),
            "expected_min_proof_windows": case.get("expected_min_proof_windows"),
            "resources": {"engine_version": index["identity"]["engine_version"], "index_id": index["index_id"],
                          "source_bytes_sha256": index["identity"]["bytes_sha256"],
                          "context_sha256": index["identity"]["context_sha256"],
                          "answering_sha256": index["answering"]["sha256"],
                          "lexicon_sha256": index["segmentation"]["lexicon"]["sha256"],
                          "wordnet_sha256": index["query_expansion"]["resource"]["sha256"]},
            "question": {"type": question["type"], "target": question["target"],
                         "keywords": question["keywords"],
                         "filtered": [d["text"] for d in question["decisions"] if d["action"] == "filtered"]},
            "answer": {key: answer[key] for key in ("status", "text", "fields", "evidence_ids", "template_id", "reason", "bindings", "evidence_group_ids")},
            "window_statistics": {"policy": index["context"]["policy"], "config": index["context"]["config"],
                                  **{key: index["context"]["statistics"][key] for key in
                                     ("N", "avgdl", "total_tokens", "unit_count", "singleton_count", "window_count")},
                                  "candidate_count": result["window_retrieval"]["candidate_count"],
                                  "searched_window_ids": result["sentence_evidence"]["searched_window_ids"]},
            "evidence_groups": [{**{key: group[key] for key in
                                   ("id", "sentence_ids", "direct_sentence_ids", "supporting_sentence_ids", "context_sentence_ids", "window_ids", "coverage", "binding", "is_answer_support")},
                                 "source_spans": [{**span, "source_text": index["source"]["text"][span["source_start"]:span["source_end"]]}
                                                  for span in group["source_spans"]]}
                                for group in groups],
            "evidence": [{key: item[key] for key in ("sentence_id", "source_text", "source_start", "source_end", "rank", "score", "complete")}
                         for item in evidence],
            "checks": checks, "passed": all(checks.values()),
        })
    positives = [row for row in records if row["expected_status"] == "answered"]
    negatives = [row for row in records if row["expected_status"] != "answered"]
    evidence_cases = [row for row in records if row["expected_evidence"] or row["expected_evidence_sentences"]]
    binding_cases = [row for row in records if row["expected_binding"] is not None]
    group_cases = [row for row in records if row["expected_group_sentence_ids"]]
    return {
        "schema": "article-qa-evaluation/2",
        "scope": "固定的短文章片段和原创测试短文；不代表任意文章、整本书或开放问题的准确率。",
        "source_catalog": catalog,
        "summary": {
            "cases": len(records), "passed": sum(row["passed"] for row in records),
            "positive_cases": len(positives), "positive_passed": sum(row["passed"] for row in positives),
            "negative_cases": len(negatives), "negative_passed": sum(row["passed"] for row in negatives),
            "expected_evidence_cases": len(evidence_cases),
            "expected_evidence_found": sum(row["checks"]["evidence_found"] for row in evidence_cases),
            "expected_binding_cases": len(binding_cases),
            "expected_binding_passed": sum(row["checks"]["binding_expected"] for row in binding_cases),
            "expected_group_cases": len(group_cases),
            "expected_group_found": sum(row["checks"]["expected_group_found"] for row in group_cases),
            "all_passed": all(row["passed"] for row in records),
        },
        "cases": records,
    }


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="运行带来源的短文章问答验收集。")
    parser.add_argument("--output", type=Path, help="将逐题检查和字段区间保存为 JSON。")
    args = parser.parse_args(argv)
    report = evaluate_articles()
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for row in report["cases"]:
        fields = ", ".join(f"{name}={field['text']}" for name, field in row["answer"]["fields"].items()) or "无字段"
        failed = [name for name, passed in row["checks"].items() if not passed]
        print(f"{'PASS' if row['passed'] else 'FAIL'} {row['id']}: {row['answer']['status']}; {fields}"
              + (f"; 未通过={','.join(failed)}" if failed else ""))
    print(json.dumps(report["summary"], ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(str(args.output.resolve()))
    return 0 if report["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
