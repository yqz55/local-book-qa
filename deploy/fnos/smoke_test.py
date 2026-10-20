"""容器内自检：核对数据可读，并用 Streamlit AppTest 实际运行一次 app.py。

结果写成 /status 下的标记文件名，便于通过文件列表远程查看。
"""

import json
import os
import traceback
from pathlib import Path

STATUS = Path("/status")
DATA = Path("/data")


def mark(name: str) -> None:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)[:180]
    (STATUS / safe).touch()


def main() -> None:
    convos = sorted((DATA / "conversations").glob("*.json"))
    indexes = sorted((DATA / "indexes").glob("*.json.gz"))
    traces = sorted((DATA / "traces").glob("*.json.gz"))

    turns = 0
    bad = 0
    for path in convos:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            turns += len(data.get("turns", []))
        except Exception:
            bad += 1
    mark(f"data-conversations-{len(convos)}-turns-{turns}-bad-{bad}")
    mark(f"data-indexes-{len(indexes)}")
    mark(f"data-traces-{len(traces)}")

    empty = Path("/tmp/nlp-empty")
    empty.mkdir(parents=True, exist_ok=True)
    os.environ["NLP_DATA_DIR"] = str(empty)
    # streamlit run 会把脚本所在目录加入 sys.path；AppTest 不会，这里手动补上。
    if "/app" not in os.sys.path:
        os.sys.path.insert(0, "/app")
    try:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file("/app/app.py", default_timeout=180)
        app.run()
        errors = [str(item.value) for item in app.exception]
        if errors:
            mark("app-EXC-" + "; ".join(errors))
        else:
            mark(f"app-OK-buttons-{len(app.button)}-markdown-{len(app.markdown)}")
    except Exception:
        mark("app-FATAL-" + traceback.format_exc(limit=2))


if __name__ == "__main__":
    main()
