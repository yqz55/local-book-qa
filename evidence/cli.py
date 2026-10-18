"""Usage: python -m evidence.cli {index,search,replay} ..."""

import argparse
import json
from pathlib import Path
import sys

from .ingest import IngestConfig
from .retrieval import BM25Config, build_index, configure_query_expansion, prepare_index_for_search, rebuild_with_book_words, search
from .storage import Store, atomic_write
from .tracing import make_trace, read_record, rebuild_index, replay, serialize

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    # Keep Chinese/emoji output stable in Windows terminals and redirected pipes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="问题提炼、句子证据、确定性规则问答与独立复算")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("index", help="导入 TXT 并保存本地索引")
    p.add_argument("file", type=Path)
    p.add_argument("--encoding", choices=["utf-8", "gb18030"], default="utf-8")
    p.add_argument("--max-chars", type=int, default=400)
    p.add_argument("--dictionary", type=Path, default=ROOT / "data" / "dictionary.json")
    p.add_argument("--book-dictionary", type=Path, help="补充本书专用词条 JSON 数组，与基础词典合并")
    p.add_argument("--aliases", type=Path, help="可选人工校正分组 JSON；默认空，姓名与关系由本书规则发现")
    p.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    p = sub.add_parser("rebuild", help="从已保存原始字节应用本书词条并重建索引，保留旧索引")
    p.add_argument("index", type=Path)
    p.add_argument("--book-dictionary", type=Path, required=True, help="新的本书专用词条 JSON 数组；空数组可清除专用词条")
    p.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    p = sub.add_parser("search", help="检索一个已保存索引")
    p.add_argument("index", type=Path)
    p.add_argument("query")
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--export", type=Path)
    p.add_argument("--no-expansion", action="store_true", help="本次检索关闭通用词义扩展，人物别名保留")
    p = sub.add_parser("configure-expansion", help="应用本书表达扩展设置，不改变正文分词")
    p.add_argument("index", type=Path)
    p.add_argument("--settings", type=Path, required=True, help="含 config、groups、blocked 的 JSON，配置可省略")
    p.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    p = sub.add_parser("replay", help="校验记录并从原始字节重建、复算全部排名")
    p.add_argument("record", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            words = json.loads(args.dictionary.read_text(encoding="utf-8-sig"))
            book_words = json.loads(args.book_dictionary.read_text(encoding="utf-8-sig")) if args.book_dictionary else []
            aliases = json.loads(args.aliases.read_text(encoding="utf-8-sig")) if args.aliases else None
            index = build_index(args.file.read_bytes(), args.file.name, args.encoding, words,
                                IngestConfig(max_chars=args.max_chars), book_words=book_words, alias_groups=aliases)
            path = Store(args.data_dir).save_index(index)
            result = {"index": str(path), "index_id": index["index_id"], "statistics": {
                k: v for k, v in index["statistics"].items() if k not in {"collection_tf", "document_frequency"}}}
        elif args.command == "rebuild":
            original = rebuild_index(read_record(args.index))
            book_words = json.loads(args.book_dictionary.read_text(encoding="utf-8-sig"))
            index = rebuild_with_book_words(original, book_words)
            path = Store(args.data_dir).save_index(index)
            result = {"index": str(path), "index_id": index["index_id"], "previous_index_id": original["index_id"],
                      "book_words": index["dictionary"]["book_words"], "N": index["statistics"]["N"]}
        elif args.command == "configure-expansion":
            original = prepare_index_for_search(rebuild_index(read_record(args.index)))
            settings = json.loads(args.settings.read_text(encoding="utf-8-sig"))
            if not isinstance(settings, dict) or set(settings) - {"config", "groups", "blocked"}:
                raise ValueError("扩展设置仅支持 config、groups、blocked 字段。")
            cfg = {**original["query_expansion"]["config"], **settings.get("config", {})}
            index = configure_query_expansion(original, config=cfg, groups=settings.get("groups"), blocked=settings.get("blocked"))
            path = Store(args.data_dir).save_index(index)
            result = {"index": str(path), "index_id": index["index_id"], "previous_index_id": original["index_id"],
                      "config": index["query_expansion"]["config"], "groups": index["query_expansion"]["manual_groups"],
                      "blocked": index["query_expansion"]["blocked"]}
        elif args.command == "search":
            index = prepare_index_for_search(rebuild_index(read_record(args.index)))
            if args.no_expansion:
                index = configure_query_expansion(index, config={**index["query_expansion"]["config"], "enabled": False})
            result = search(index, args.query, BM25Config(k1=args.k1, b=args.b))
            if args.export:
                trace = make_trace(index, result)
                atomic_write(args.export, serialize(trace, compress=args.export.suffix.lower() == ".gz"))
            paragraphs = {p["id"]: p for p in index["paragraphs"]}
            result = {"query": result["query"], "status": result["status"],
                      "question": result.get("question"),
                      "sentence_evidence": result.get("sentence_evidence"),
                      "window_retrieval": result.get("window_retrieval"),
                      "evidence_groups": result.get("evidence_groups"),
                      "answer": result.get("answer"),
                      "general_answer_policy": result.get("general_answer_policy"),
                      "query_tokens": result["query_tokens"], "query_tokenization": result["query_tokenization"],
                      "query_expansions": result.get("query_expansions", []),
                      "lexical_expansions": result.get("lexical_expansions", []),
                      "lexical_decisions": result.get("lexical_decisions", []),
                      "query_expansion_policy": result.get("query_expansion_policy"),
                      "query_ambiguities": result.get("query_ambiguities", []),
                      "candidate_count": result["candidate_count"],
                      "top": [{**c, "text": paragraphs[c["paragraph_id"]]["text"]}
                              for c in result["candidates"][:5]], "export": str(args.export) if args.export else None}
        else:
            result = replay(read_record(args.record))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError, EOFError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
