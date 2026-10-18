"""Replay the finite source-bound reason/method article acceptance set offline.

Run ``python -m evidence.evaluate_discourse --output .data/discourse-evaluation.json``.
The authored cases are implementation acceptance examples, not a held-out QA
benchmark. Source titles, dates and attribution never enter the indexed body.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from .retrieval import build_index, search


ARTICLE_DIR = Path(__file__).resolve().parents[1] / "data" / "discourse_articles"
SPAN_KEYS = {"text", "source_start", "source_end", "clean_start", "clean_end", "sentence_id", "rule"}
ANSWER_STATES = {"answered", "evidence_only", "ambiguous", "unsupported", "insufficient_evidence", "no_content"}
REFUSAL_STATES = {"evidence_only", "unsupported", "insufficient_evidence"}


def load_articles(folder=ARTICLE_DIR):
    """Load only declared fixture bytes and reject missing or changed snapshots."""
    folder = Path(folder)
    catalog = json.loads((folder / "sources.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
    sources = catalog["sources"]
    names = [source["fixture"] for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("文章来源清单有重复文件名。")
    payloads = {}
    for source in sources:
        name = source["fixture"]
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("文章文件名不得含目录。")
        payload = (folder / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != source["text_sha256"] or len(payload) != source["byte_length"]:
            raise ValueError(f"文章快照字节与来源清单不一致：{name}")
        payload.decode("utf-8", errors="strict")
        payloads[name] = payload
    cases = manifest["cases"]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("问题清单有重复编号。")
    for case in cases:
        if case["fixture"] not in payloads:
            raise ValueError(f"测试使用未注明来源的文章：{case['fixture']}")
    return catalog, cases, payloads


def _nested_spans(value):
    """Visit actual span records without assuming a particular binding shape."""
    if isinstance(value, dict):
        if "text" in value and ("source_start" in value or "clean_start" in value):
            yield value
        for child in value.values():
            yield from _nested_spans(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _nested_spans(child)


def _valid_span(span, index, sentences):
    if not SPAN_KEYS.issubset(span):
        return False
    coordinates = [span[name] for name in ("source_start", "source_end", "clean_start", "clean_end")]
    if any(type(value) is not int for value in coordinates):
        return False
    sa, sb, ca, cb = coordinates
    sentence = sentences.get(span["sentence_id"])
    return bool(sentence and span["text"] and span["rule"]
                and 0 <= sa < sb <= len(index["source"]["text"])
                and 0 <= ca < cb <= len(index["cleaning"]["text"])
                and sentence["source_start"] <= sa < sb <= sentence["source_end"]
                and sentence["clean_start"] <= ca < cb <= sentence["clean_end"]
                and index["source"]["text"][sa:sb] == span["text"]
                and index["cleaning"]["text"][ca:cb] == span["text"])


def _template_matches(index, question, answer):
    template = index["answering"]["templates"].get(question["type"])
    if not template or template["id"] != answer.get("template_id"):
        return False
    fields = answer["fields"]
    if not all(name in fields for name in template["required"]):
        return False
    try:
        return answer["text"] == template["text"].format(**{name: field["text"] for name, field in fields.items()})
    except (KeyError, ValueError, TypeError):
        return False


def evaluate_articles(folder=ARTICLE_DIR):
    """Return per-case state, field, proof, mapping and template observations."""
    catalog, cases, payloads = load_articles(folder)
    indexes, records = {}, []
    for case in cases:
        name = case["fixture"]
        if name not in indexes:
            indexes[name] = build_index(payloads[name], name, "utf-8", [])
        index = indexes[name]
        result = search(index, case["query"])
        answer, question = result["answer"], result["question"]
        evidence = result["sentence_evidence"]["candidates"]
        sentences = {row["sentence_id"]: row for row in evidence}
        groups = result["evidence_groups"]["groups"]
        by_group = {group["id"]: group for group in groups}
        answer_groups = [by_group[group_id] for group_id in answer.get("evidence_group_ids", []) if group_id in by_group]
        accepted = [decision for decision in answer.get("decisions", []) if decision.get("accepted")]
        expected = case["expected_status"]
        checks = {
            "known_answer_state": answer["status"] in ANSWER_STATES,
            "answer_state": answer["status"] in REFUSAL_STATES if expected == "not_answered" else answer["status"] == expected,
            "field_source_spans": all(_valid_span(field, index, sentences) for field in answer["fields"].values()),
            "decision_source_spans": all(_valid_span(span, index, sentences) for span in _nested_spans(answer.get("decisions", []))),
            "binding_source_spans": all(_valid_span(span, index, sentences) for span in _nested_spans(
                [answer.get("bindings", []), *[group.get("binding") for group in groups]])),
            "groups_deduplicated": len(groups) == len({tuple(group["sentence_ids"]) for group in groups}),
            "group_source_spans": all(
                span["sentence_id"] in sentences
                and all(span[key] == sentences[span["sentence_id"]][key]
                        for key in ("source_start", "source_end", "clean_start", "clean_end"))
                and index["source"]["text"][span["source_start"]:span["source_end"]] == sentences[span["sentence_id"]]["source_text"]
                for group in groups for span in group["source_spans"]),
            "group_sentence_coverage": all(
                group["sentence_ids"] == [span["sentence_id"] for span in group["source_spans"]]
                and set(group["sentence_ids"]).issubset(sentences)
                and all(span["role"] in {"direct", "support", "context"} for span in group["source_spans"])
                for group in groups),
            "group_roles_consistent": all(
                {span["sentence_id"] for span in group["source_spans"] if span["role"] == "direct"} == set(group["direct_sentence_ids"])
                and {span["sentence_id"] for span in group["source_spans"] if span["role"] == "support"} == set(group["supporting_sentence_ids"])
                and {span["sentence_id"] for span in group["source_spans"] if span["role"] == "context"} == set(group["context_sentence_ids"])
                for group in groups),
            "question_words_filtered": not set(question["keywords"]).intersection(
                {"为什么", "为何", "怎样", "如何", "怎么", "怎样才能", "如何才能", "怎么才能",
                 "什么原因", "是什么原因", "用什么方法", "通过什么方式", "才能"}),
        }
        if expected != "answered":
            checks["no_unsupported_fields"] = not answer["fields"]
        for field_name, text in case.get("expected_fields", {}).items():
            checks["field_value_" + field_name] = answer["fields"].get(field_name, {}).get("text") == text
        if case.get("expected_evidence_sentences"):
            checks["evidence_found"] = all(any(row["source_text"] == text for row in evidence)
                                           for text in case["expected_evidence_sentences"])
        if case.get("expected_statement_kind"):
            checks["statement_kind"] = answer.get("statement_kind") == case["expected_statement_kind"]
        if "expected_binding" in case:
            checks["binding_expected"] = bool(answer.get("bindings")) == case["expected_binding"]
            if case["expected_binding"]:
                checks["binding_has_real_spans"] = bool(answer.get("bindings")) and all(
                    len(list(_nested_spans(binding))) >= 2 for binding in answer.get("bindings", []))
        if case.get("expected_group"):
            wanted = case["expected_group"]
            matching = [group for group in answer_groups if group["sentence_ids"] == wanted["sentence_ids"]]
            checks["expected_group_found"] = len(matching) == 1 and all(
                matching[0][role] == wanted[role] for role in ("direct_sentence_ids", "supporting_sentence_ids"))
        if case.get("expected_alternatives"):
            for field_name, values in case["expected_alternatives"].items():
                observed = {decision.get("fields", {}).get(field_name, {}).get("text") for decision in accepted}
                checks["alternatives_" + field_name] = observed == set(values)
        if case.get("expected_min_answer_groups"):
            checks["alternative_proof_groups"] = len(answer_groups) >= case["expected_min_answer_groups"]
        if expected == "answered":
            checks["answer_has_evidence"] = bool(answer["evidence_ids"]) and set(answer["evidence_ids"]).issubset(sentences)
            checks["answer_has_proof_groups"] = bool(answer_groups) and all(group["is_answer_support"] for group in answer_groups)
            proof_ids = {sid for group in answer_groups for sid in group["sentence_ids"]}
            checks["fields_in_answer_proof"] = all(field["sentence_id"] in proof_ids for field in answer["fields"].values())
            checks["accepted_decision_matches"] = any(all(
                decision.get("fields", {}).get(field_name) == field for field_name, field in answer["fields"].items()) for decision in accepted)
            checks["source_bound_template"] = _template_matches(index, question, answer)
        records.append({
            "id": case["id"], "fixture": name, "query": case["query"], "purpose": case["purpose"],
            "expected_status": expected, "expected_fields": case.get("expected_fields", {}),
            "expected_evidence_sentences": case.get("expected_evidence_sentences", []),
            "expected_binding": case.get("expected_binding"), "expected_group": case.get("expected_group"),
            "expected_alternatives": case.get("expected_alternatives"),
            "expected_statement_kind": case.get("expected_statement_kind"),
            "resources": {"engine_version": index["identity"]["engine_version"], "index_id": index["index_id"],
                          "source_bytes_sha256": index["identity"]["bytes_sha256"],
                          "context_sha256": index["identity"]["context_sha256"],
                          "answering_sha256": index["answering"]["sha256"],
                          "lexicon_sha256": index["segmentation"]["lexicon"]["sha256"],
                          "wordnet_sha256": index["query_expansion"]["resource"]["sha256"]},
            "question": {key: question[key] for key in ("type", "status", "target", "keywords", "constraints", "decisions")},
            "answer": {**{key: answer[key] for key in ("status", "text", "fields", "evidence_ids", "template_id", "reason", "bindings", "evidence_group_ids", "decisions")},
                       "template": answer.get("template"), "statement_kind": answer.get("statement_kind"),
                       "rule_ids": answer.get("rule_ids", [])},
            "window_statistics": {**index["context"]["statistics"],
                                  "searched_window_ids": result["sentence_evidence"]["searched_window_ids"]},
            "evidence": [{key: row[key] for key in ("sentence_id", "source_text", "source_start", "source_end", "clean_start", "clean_end", "rank", "score", "complete")} for row in evidence],
            "evidence_groups": [{**group, "source_spans": [{**span, "source_text": index["source"]["text"][span["source_start"]:span["source_end"]]} for span in group["source_spans"]]} for group in groups],
            "checks": checks, "passed": all(checks.values()),
        })
    positives = [row for row in records if row["expected_status"] == "answered"]
    negatives = [row for row in records if row["expected_status"] != "answered"]
    evidence_rows = [row for row in records if row["expected_evidence_sentences"]]
    return {
        "schema": "discourse-article-evaluation/1",
        "scope": "有限的短文章原因、方法规则验收；不代表任意文章、整本书或开放问题的准确率。",
        "source_catalog": catalog,
        "summary": {"cases": len(records), "passed": sum(row["passed"] for row in records),
                    "positive_cases": len(positives), "positive_passed": sum(row["passed"] for row in positives),
                    "negative_cases": len(negatives), "negative_passed": sum(row["passed"] for row in negatives),
                    "expected_evidence_cases": len(evidence_rows),
                    "expected_evidence_found": sum(row["checks"]["evidence_found"] for row in evidence_rows),
                    "observed_states": dict(Counter(row["answer"]["status"] for row in records)),
                    "all_passed": all(row["passed"] for row in records)},
        "cases": records,
    }


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="运行离线短文章原因／方法问答验收。")
    parser.add_argument("--fixtures", type=Path, default=ARTICLE_DIR, help="带 sources.json 和 cases.json 的材料目录。")
    parser.add_argument("--output", type=Path, help="保存逐题证据、模板、来源区间和检查结果为 JSON。")
    args = parser.parse_args(argv)
    report = evaluate_articles(args.fixtures)
    for row in report["cases"]:
        fields = ", ".join(f"{name}={field['text']}" for name, field in row["answer"]["fields"].items()) or "无字段"
        failed = [name for name, passed in row["checks"].items() if not passed]
        print(f"{'PASS' if row['passed'] else 'FAIL'} {row['id']}: {row['answer']['status']}; {fields}"
              + (f"; 未通过={','.join(failed)}" if failed else ""))
    print(json.dumps(report["summary"], ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(str(args.output.resolve()))
    return 0 if report["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
