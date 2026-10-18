import json
from pathlib import Path
import subprocess
import sys

from streamlit.testing.v1 import AppTest

from evidence.retrieval import build_index
from evidence.storage import Store
from ui.answers import sentence_quote


def test_raw_sentence_quote_escapes_markup_and_maps_answer_fields():
    raw = "导语。\n雨水是 <script> 里的词。\n"
    start = raw.index("雨水")
    end = raw.index("。", start) + 1
    left = raw.index("<script>")
    candidate = {"source_start": start, "source_end": end, "text": "不能用清洗后的展示替代原文"}
    fields = {"definition": {"text": "<script>", "source_start": left, "source_end": left + 8}}
    rendered = sentence_quote({"source": {"text": raw}}, candidate, fields)
    assert rendered == "雨水是 <mark>&lt;script&gt;</mark> 里的词。"
    assert "不能用" not in rendered
    assert "<script>" not in rendered


def test_cli_search_preserves_retrieval_and_includes_answer_snapshots(tmp_path):
    index = build_index("雨水收集是将降雨储存后用于灌溉的方法。".encode("utf-8"),
                        "雨水短文.txt", "utf-8", [], book_words=["雨水收集"])
    path = Store(tmp_path).save_index(index)
    process = subprocess.run([sys.executable, "-m", "evidence.cli", "search", str(path), "雨水收集是什么？"],
                             cwd=Path(__file__).resolve().parents[1], capture_output=True, encoding="utf-8", timeout=60)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert {"query", "status", "query_tokens", "query_tokenization", "query_expansions", "lexical_expansions",
            "lexical_decisions", "query_expansion_policy", "query_ambiguities", "candidate_count", "top", "export"} <= result.keys()
    assert result["question"]["keywords"] == ["雨水收集"]
    assert result["sentence_evidence"]["candidates"]
    assert result["answer"]["status"] == "answered"
    assert result["top"][0]["text"] == index["paragraphs"][0]["text"]


def test_short_article_question_answer_and_unsupported_state(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_DATA_DIR", str(tmp_path))
    article = "雨水收集\n雨水收集是将降雨储存后用于灌溉的方法。\n示范项目于2025年启动。\n储水箱容量为300升。"
    index = build_index(article.encode("utf-8"), "短文·雨水收集.txt", "utf-8", [], book_words=["雨水收集"])
    Store(tmp_path).save_index(index)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui" / "workbench.py"), default_timeout=60).run()
    assert not app.exception

    def ask(query):
        next(t for t in app.text_input if t.label == "输入问题或关键词").set_value(query)
        next(b for b in app.button if b.label == "检索").click().run()
        assert not app.exception
        return app.session_state["result"]

    result = ask("雨水收集是什么？")
    assert result["answer"]["status"] == "answered"
    assert any("已形成规则答案" in alert.value for alert in app.success)
    assert any("实际检索关键词：" in text.value and "雨水收集" in text.value for text in app.text)
    assert any("规则答案与句子证据" in heading.value for heading in app.subheader)
    assert any("雨水收集是将降雨储存后用于灌溉的方法。" in block.value for block in app.code)
    assert any("<mark>" in block.value for block in app.markdown)
    assert len(app.tabs) == 3
    assert [tab.label for tab in app.tabs] == ["检索证据", "文本与索引", "记录复算"]

    result = ask("为什么要雨水收集？")
    assert result["answer"]["status"] == "evidence_only"
    assert result["answer"]["mode"] == "reference"
    assert "雨水收集" in result["answer"]["text"]
    assert "原因" in result["answer"]["reason"]
    assert any("当前问法暂不支持规则作答" in alert.value for alert in app.info)
    assert result["sentence_evidence"]["candidates"]

    result = ask("为什么会这样？")
    assert result["answer"]["status"] == "no_content"
    assert any("问题缺少可检索的内容" in alert.value for alert in app.info)

