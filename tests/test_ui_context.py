import json
from pathlib import Path
import subprocess
import sys

from streamlit.testing.v1 import AppTest

from evidence.retrieval import build_index
from evidence.storage import Store
from ui.answers import sentence_quote
from ui.context_evidence import group_sentence_spans, visible_groups


def grouped_fixture():
    raw = "青禾项目进入实施阶段。\r\n该项目于2025年3月12日启动。\r\n另一项目暂停。\r\n容量为<300>升。"
    texts = ["青禾项目进入实施阶段。", "该项目于2025年3月12日启动。", "另一项目暂停。", "容量为<300>升。"]
    sentences = []
    for number, text in enumerate(texts, 1):
        start = raw.index(text)
        sentences.append({"id": f"S{number}", "text": text, "source_start": start, "source_end": start + len(text),
                          "location": {"start_line": number, "start_column": 1}, "chapter": "短文"})
    first = {"id": "G1", "sentence_ids": ["S1", "S2"], "anchor_sentence_ids": ["S2"],
             "supporting_sentence_ids": ["S1"], "context_sentence_ids": [], "window_ids": ["W1"],
             "source_spans": [{**record, "sentence_id": record["id"],
                               "role": "support" if record["id"] == "S1" else "direct",
                               "reason": "明确的前句对象" if record["id"] == "S1" else "包含被问时间"}
                              for record in sentences[:2]]}
    second = {"id": "G2", "sentence_ids": ["S2", "S4"], "context_sentence_ids": ["S4"],
              "source_spans": [{**sentences[1], "sentence_id": "S2", "role": "direct", "reason": "时间命中"},
                               {**sentences[3], "sentence_id": "S4", "role": "context", "reason": "仅作补充"}]}
    value_start = raw.index("2025")
    result = {"question": {"type": "time", "target": "青禾项目", "keywords": ["青禾项目", "启动"]},
              "sentence_evidence": {"candidates": [{**record, "sentence_id": record["id"]} for record in sentences]},
              "answer": {"status": "answered", "text": "原文记载时间为2025年3月12日。", "evidence_ids": ["S2"],
                         "evidence_group_ids": ["G1"], "fields": {"time": {"source_start": value_start,
                                                                                "source_end": value_start + len("2025年3月12日")}}},
              "evidence_groups": {"groups": [second, first, {**first, "id": "G3"}], "top_ids": ["G2", "G1", "G3"]}}
    return {"source": {"text": raw}, "sentences": {"sentences": sentences}}, result


def test_answer_groups_prioritized_without_repeating_overlapping_quotations():
    _, result = grouped_fixture()
    groups = visible_groups(result)
    assert [group["id"] for group in groups] == ["G1", "G2"]
    assert groups[0]["display_sentence_ids"] == ["S1", "S2"]
    assert groups[1]["display_sentence_ids"] == ["S4"]


def test_group_quotes_keep_roles_and_literal_disjoint_source_ranges():
    index, result = grouped_fixture()
    group = visible_groups(result)[1]
    spans = group_sentence_spans(index, result, group)
    assert len(spans) == 1
    assert spans[0]["role"] == "context"
    assert sentence_quote(index, spans[0], {}) == "容量为&lt;300&gt;升。"
    assert "另一项目" not in sentence_quote(index, spans[0], {})
    assert result["evidence_groups"]["groups"][0]["sentence_ids"] == ["S2", "S4"]


def test_multi_sentence_cards_explain_roles_and_keep_individual_source_quotes():
    index, result = grouped_fixture()
    code = ("import json\nfrom ui.answers import show_answer\n"
            f"index = json.loads({json.dumps(index)!r})\n"
            f"result = json.loads({json.dumps(result)!r})\n"
            "show_answer(index, result)\n")
    app = AppTest.from_string(code, default_timeout=30).run()
    assert not app.exception
    captions = [element.value for element in app.caption]
    assert any("关联证据 · S1" in value for value in captions)
    assert any("直接证据 · S2" in value for value in captions)
    assert any("补充上下文 · S4" in value for value in captions)
    assert any("加入原因：明确的前句对象" == value for value in captions)
    quotes = [element.value for element in app.code]
    assert quotes == ["青禾项目进入实施阶段。", "该项目于2025年3月12日启动。", "容量为<300>升。"]
    assert any("<mark>2025年3月12日</mark>" in element.value for element in app.markdown)
    assert not any("另一项目暂停" in element.value for element in app.markdown)


def test_window_diagnostics_display_independent_statistics_and_per_sentence_quotes():
    index, result = grouped_fixture()
    result["corpus"] = {"N": 99, "avgdl": 100.0}
    result["window_retrieval"] = {
        "corpus": {"N": 7, "avgdl": 12.5}, "config": {"k1": 1.5, "b": 0.75},
        "candidate_count": 1, "query_terms": [{"term": "青禾项目", "df": 3}],
        "candidates": [{"window_id": "W1", "score": 1.25, "sentence_ids": ["S1", "S2"],
                        "paragraph_ids": ["P1", "P2"], "dl": 20,
                        "term_details": [{"term": "青禾项目", "matched": True, "tf": 1,
                                          "df": 3, "idf": 0.8, "contribution": 1.25}]}]}
    code = ("import json\nfrom ui.context_evidence import show_window_retrieval\n"
            f"index = json.loads({json.dumps(index)!r})\n"
            f"result = json.loads({json.dumps(result)!r})\n"
            "show_window_retrieval(index, result)\n")
    app = AppTest.from_string(code, default_timeout=30).run()
    assert not app.exception
    captions = [element.value for element in app.caption]
    assert any("N = 7" in value and "avgdl = 12.500000" in value for value in captions)
    assert not any("N = 99" in value for value in captions)
    assert list(app.dataframe[0].value["窗口 DF"]) == [3]
    assert [element.value for element in app.code[:2]] == ["青禾项目进入实施阶段。", "该项目于2025年3月12日启动。"]


def test_cross_paragraph_answer_app_and_cli_expose_same_evidence_groups(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_DATA_DIR", str(tmp_path))
    article = "青禾社区雨水收集项目进入实施阶段。\n该项目于2025年3月12日启动。\n储水箱容量为300升。"
    index = build_index(article.encode("utf-8"), "项目短文.txt", "utf-8", [])
    path = Store(tmp_path).save_index(index)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui" / "workbench.py"), default_timeout=60).run()
    assert not app.exception
    query = "青禾社区雨水收集项目什么时候启动？"
    next(widget for widget in app.text_input if widget.label == "输入问题或关键词").set_value(query)
    next(widget for widget in app.button if widget.label == "检索").click().run()
    assert not app.exception
    result = app.session_state["result"]
    assert result["answer"]["status"] == "answered"
    assert result["answer"]["evidence_group_ids"]
    assert any("关联证据 ·" in widget.value for widget in app.caption)
    assert any("直接证据 ·" in widget.value for widget in app.caption)
    assert any("句子窗口检索" == widget.value for widget in app.subheader)
    assert any("原始段落检索诊断与记录导出" == widget.label for widget in app.expander)

    command = [sys.executable, "-m", "evidence.cli", "search", str(path), query]
    process = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True,
                             encoding="utf-8", timeout=60)
    assert process.returncode == 0, process.stderr
    output = json.loads(process.stdout)
    assert output["window_retrieval"] == result["window_retrieval"]
    assert output["evidence_groups"] == result["evidence_groups"]
    assert output["answer"] == result["answer"]
    assert output["top"]  # Legacy paragraph diagnostic output stays available.
