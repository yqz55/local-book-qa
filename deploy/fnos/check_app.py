"""容器内端到端自检：用真实数据目录运行一次 app.py，确认历史对话能正常打开。"""

import os
import sys
from pathlib import Path

STATUS = Path("/status")

sys.path.insert(0, "/app")
os.environ["NLP_DATA_DIR"] = "/data"


def mark(name: str) -> None:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)[:180]
    (STATUS / safe).touch()


def main() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("/app/app.py", default_timeout=300)
    app.run()

    errors = [str(item.value) for item in app.error]
    exceptions = [str(item.value) for item in app.exception]
    if exceptions:
        mark("app-real-FATAL-" + "; ".join(exceptions))
        return
    if errors:
        mark("app-real-ERROR-" + "; ".join(errors))
        return
    mark(f"app-real-OK-chatinput-{len(app.chat_input)}-buttons-{len(app.button)}-markdown-{len(app.markdown)}")


if __name__ == "__main__":
    main()
