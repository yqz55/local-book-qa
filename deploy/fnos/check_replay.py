"""容器内复算自检：对每一条已保存对话重新建索引并复算最后一条问答记录。

用 Python 3.13 运行（Unicode 15.1.0），与生成记录时的版本一致：
- 通过 → 生成 replay-ok-<n> 标记；
- 失败 → 生成 replay-fail-<原因> 标记（文件名里带失败原因）。
"""

import json
import os
import sys
from pathlib import Path

STATUS = Path("/status")
DATA = Path("/data")

sys.path.insert(0, "/app")
os.environ.setdefault("NLP_DATA_DIR", "/tmp/nlp-check-rw")
Path(os.environ["NLP_DATA_DIR"]).mkdir(parents=True, exist_ok=True)


def mark(name: str) -> None:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)[:180]
    (STATUS / safe).touch()


def main() -> None:
    import unicodedata

    from evidence.retrieval import prepare_index_for_search
    from evidence.tracing import read_record, rebuild_index, replay

    mark(f"check-python-{sys.version_info.major}.{sys.version_info.minor}-unicode-{unicodedata.unidata_version}")

    ok = 0
    fails = []
    for conv_path in sorted((DATA / "conversations").glob("*.json")):
        conv = json.loads(conv_path.read_text(encoding="utf-8"))
        index_id = conv["index_id"]
        try:
            rebuilt = rebuild_index(read_record(str(DATA / "indexes" / f"{index_id}.json.gz")))
            prepare_index_for_search(rebuilt)
            if rebuilt["index_id"] != index_id:
                fails.append(f"{index_id[:8]}-id-mismatch")
                continue
            turns = conv.get("turns") or []
            if turns:
                replay(read_record(str(DATA / "traces" / f"{turns[-1]['trace_id']}.json.gz")))
            ok += 1
        except Exception as exc:  # noqa: BLE001 - 需要把原因带出来
            fails.append(f"{index_id[:8]}:{type(exc).__name__}:{exc}"[:150])

    mark(f"replay-ok-{ok}")
    if fails:
        mark("replay-fail-" + ";".join(fails))


if __name__ == "__main__":
    main()
