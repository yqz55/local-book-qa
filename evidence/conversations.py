"""Small, durable chat records; full algorithm snapshots stay in the trace store."""

from copy import deepcopy
from datetime import datetime, timezone
import re
from uuid import uuid4

from .retrieval import canonical
from .storage import Store, atomic_write
from .tracing import read_record


def now():
    return datetime.now(timezone.utc).isoformat()


def _identifier(value, length):
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError("记录标识无效。")
    return value


class ConversationStore:
    """A new UUID per upload, even when the underlying index is identical."""

    def __init__(self, store: Store):
        self.store = store
        self.root = store.root / "conversations"

    def _path(self, conversation_id):
        return self.root / f"{_identifier(conversation_id, 32)}.json"

    def load(self, conversation_id):
        value = read_record(self._path(conversation_id))
        if value.get("schema") != "evidence-conversation/1" or value.get("id") != conversation_id:
            raise ValueError("对话记录格式无效。")
        _identifier(value["index_id"], 64)
        if not isinstance(value.get("turns"), list):
            raise ValueError("对话消息格式无效。")
        return value

    def save(self, conversation):
        atomic_write(self._path(conversation["id"]), canonical(conversation))

    def list(self):
        records = []
        for path in self.root.glob("*.json"):
            try:
                value = self.load(path.stem)
                records.append({key: value[key] for key in
                                ("id", "title", "filename", "index_id", "created_at", "updated_at")})
            except (ValueError, OSError, KeyError, TypeError):
                continue
        return sorted(records, key=lambda value: (value["updated_at"], value["id"]), reverse=True)

    def create(self, index):
        self.store.save_index(index)
        stamp = now()
        filename = index["source"]["filename"]
        conversation = {"schema": "evidence-conversation/1", "id": uuid4().hex,
                        "title": filename.rsplit(".", 1)[0] or filename, "filename": filename,
                        "index_id": index["index_id"], "created_at": stamp, "updated_at": stamp,
                        "turns": []}
        self.save(conversation)
        return conversation

    def append(self, conversation_id, trace, trace_path, process):
        conversation = self.load(conversation_id)
        if trace["index"]["index_id"] != conversation["index_id"]:
            raise ValueError("当前文档已变化，请重新提问。")
        result = trace["retrieval"]
        trace_id = trace_path.name.removesuffix(".json.gz")
        _identifier(trace_id, 64)
        conversation["turns"].append({"id": uuid4().hex, "query": result["query"],
                                      "answer": deepcopy(result["answer"]), "process": process,
                                      "index_id": conversation["index_id"], "trace_id": trace_id,
                                      "engine_version": trace["engine_version"],
                                      "created_at": trace["created_at"]})
        conversation["updated_at"] = now()
        self.save(conversation)
        return conversation

    def rename(self, conversation_id, title):
        title = title.strip()
        if not title or len(title) > 100:
            raise ValueError("对话名称需为 1–100 个字符。")
        value = self.load(conversation_id)
        value.update(title=title, updated_at=now())
        self.save(value)
        return value

    def clear(self, conversation_id):
        value = self.load(conversation_id)
        value.update(turns=[], updated_at=now())
        self.save(value)
        return value

    def change_index(self, conversation_id, index):
        value = self.load(conversation_id)
        value.update(index_id=index["index_id"], filename=index["source"]["filename"], updated_at=now())
        self.save(value)
        return value

    def delete(self, conversation_id):
        self._path(conversation_id).unlink(missing_ok=True)

    def trace_path(self, trace_id):
        return self.store.root / "traces" / f"{_identifier(trace_id, 64)}.json.gz"


def export_conversation(conversation):
    """Plain text cannot execute Markdown/HTML from a supplied document."""
    lines = [conversation["title"], f"文档：{conversation['filename']}",
             "每个问题独立检索文档，不使用前文推断。", ""]
    for number, turn in enumerate(conversation["turns"], 1):
        lines.extend((f"问题 {number}：{turn['query']}", "",
                      turn["answer"].get("text") or turn["answer"].get("reason") or "未找到可核对的答案。",
                      "", f"算法版本：{turn['engine_version']}", f"复算记录：{turn['trace_id']}", ""))
    return "\n".join(lines).encode("utf-8-sig")
