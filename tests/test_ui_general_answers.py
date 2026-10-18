"""The UI must render a direct answer and a distinct source reference."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from evidence.query_expansion import resource_snapshot
from evidence.retrieval import build_index
from evidence.storage import Store


def test_general_answer_reference_and_opinion_proof_render(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_DATA_DIR", str(tmp_path))
    text = "我们的战士，我感到他们是最可爱的人。\n研究所发布了观测报告。"
    index = build_index(text.encode(), "通用问答界面验收.txt", "utf-8", [],
                        expansion_resource=resource_snapshot([], {"name": "ui-general"}))
    Store(tmp_path).save_index(index)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui" / "workbench.py"), default_timeout=60).run()
    assert not app.exception
    next(r for r in app.radio if r.label == "结果展示方式").set_value("当前版").run()

    def ask(query):
        next(t for t in app.text_input if t.label == "输入问题或关键词").set_value(query)
        next(b for b in app.button if b.label == "检索").click().run()
        assert not app.exception
        return app.session_state["result"]["answer"]

    answer = ask("作者认为谁是最可爱的人？")
    assert answer["status"] == "answered"
    assert answer["fields"]["value"]["text"] == "我们的战士"
    assert any("回答与原文证据" in row.value for row in app.subheader)
    assert any("我们的战士" in row.value for row in app.text)
    assert any("作者在文中的观点" in row.value for row in app.caption)
    assert any("<mark>我们的战士</mark>" in row.value for row in app.markdown)

    answer = ask("研究所的工作有什么启示？")
    assert answer["mode"] == "reference"
    assert any(row.value == "原文参考回答" for row in app.info)
    assert any("研究所发布了观测报告。" in row.value and "原文参考" in row.value for row in app.text)
    assert not app.session_state["result"]["answer"]["fields"]
