"""Local atomic persistence. Filenames are generated from hashes, never uploads."""

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile

from .retrieval import canonical, digest
from .tracing import read_record, serialize


def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save_index(self, index: dict) -> Path:
        path = self.root / "indexes" / f"{index['index_id']}.json.gz"
        atomic_write(path, serialize(index))
        metadata = {"index_id": index["index_id"], "filename": index["source"]["filename"],
                    "N": index["statistics"]["N"], "saved_at": datetime.now(timezone.utc).isoformat()}
        atomic_write(path.with_suffix(".meta.json"), canonical(metadata))
        return path

    def list_indexes(self) -> list[dict]:
        return self._list_metadata("indexes")

    def list_traces(self) -> list[dict]:
        return self._list_metadata("traces")

    def _list_metadata(self, folder: str) -> list[dict]:
        entries = []
        for path in (self.root / folder).glob("*.meta.json"):
            try:
                metadata = read_record(path)
                data_path = path.with_name(path.name.removesuffix(".meta.json") + ".gz")
                if data_path.is_file():
                    entries.append({**metadata, "path": str(data_path)})
            except (ValueError, OSError):
                continue
        return sorted(entries, key=lambda e: e["saved_at"], reverse=True)

    def save_trace(self, trace: dict, data: bytes | None = None) -> Path:
        key = digest(trace)
        path = self.root / "traces" / f"{key}.json.gz"
        atomic_write(path, data if data is not None else serialize(trace))
        metadata = {"trace_id": key, "query": trace["retrieval"]["query"],
                    "engine_version": trace["engine_version"],
                    "filename": trace["index"]["source"]["filename"],
                    "saved_at": trace["created_at"], "top_ids": trace["retrieval"]["top_ids"]}
        atomic_write(path.with_suffix(".meta.json"), canonical(metadata))
        return path
