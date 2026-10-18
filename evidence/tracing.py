"""Self-contained snapshots, integrity checks, and complete recomputation."""

import base64
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path

from . import ALIAS_ENGINE_VERSION, ENGINE_VERSION, DISCOURSE_ENGINE_VERSION, WINDOW_ENGINE_VERSION, ANSWER_ENGINE_VERSION, EXPANSION_ENGINE_VERSION, DAG_ENGINE_VERSION, PERSON_ENGINE_VERSION, FREQUENCY_ENGINE_VERSION, LEGACY_ENGINE_VERSION, SUPPORTED_ENGINE_VERSIONS, schema_version
from .aliases import CURRENT_QUERY_POLICY, LEGACY_ALIAS_QUERY_POLICY
from .dag_tokenization import DAG_POLICY
from .ingest import IngestConfig
from .retrieval import BM25Config, build_index, canonical, digest, prepare_index_for_search, search
from .tokenization import QUERY_POLICY


def make_trace(index: dict, result: dict) -> dict:
    policy = result.get("query_tokenization", {}).get("policy")
    if policy == DAG_POLICY:
        if "query_expansion_policy" in result:
            if "question" in result:
                if "window_retrieval" in result:
                    if "general_answer_policy" in result:
                        engine = ENGINE_VERSION
                        index = prepare_index_for_search(index)
                    elif "discourse_policy" in result:
                        engine = DISCOURSE_ENGINE_VERSION
                        from .legacy_discourse_retrieval import prepare_index_for_search as prepare_discourse_index
                        index = prepare_discourse_index(index)
                    else:
                        engine = WINDOW_ENGINE_VERSION
                        from .legacy_window_retrieval import prepare_index_for_search as prepare_window_index
                        index = prepare_window_index(index)
                else:
                    engine = ANSWER_ENGINE_VERSION
                    from .legacy_answer_retrieval import prepare_index_for_search as prepare_answer_index
                    index = prepare_answer_index(index)
            else:
                engine = EXPANSION_ENGINE_VERSION
                from .legacy_expansion_retrieval import prepare_index_for_search as prepare_expansion_index
                index = prepare_expansion_index(index)
        else:
            engine = DAG_ENGINE_VERSION
            from .legacy_dag_retrieval import prepare_index_for_search as prepare_dag_index
            index = prepare_dag_index(index)
    elif policy == CURRENT_QUERY_POLICY:
        engine = PERSON_ENGINE_VERSION
        from .legacy_retrieval import prepare_index_for_search as prepare_person_index
        index = prepare_person_index(index)
    elif policy == LEGACY_ALIAS_QUERY_POLICY:
        engine = ALIAS_ENGINE_VERSION
    elif policy == QUERY_POLICY:
        engine = FREQUENCY_ENGINE_VERSION
    elif policy is None:
        engine = LEGACY_ENGINE_VERSION
    else:
        raise ValueError("不支持该查询分词策略。")
    if result["corpus"]["index_id"] != index["index_id"]:
        raise ValueError("问题与索引版本不一致。")
    payload = {"schema": f"evidence-trace/{schema_version(engine)}", "engine_version": engine,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "position_convention": "Unicode code points, 0-based half-open offsets; displayed lines/columns are 1-based",
               "index": index, "retrieval": result}
    return {**payload, "integrity_sha256": digest(payload)}


def serialize(value: dict, compress: bool = True) -> bytes:
    data = canonical(value)
    return gzip.compress(data, mtime=0) if compress else data


def deserialize(data: bytes) -> dict:
    if data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)

    def no_constant(value):
        raise ValueError(f"记录含非有限数字：{value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON 有重复字段：{key}")
            result[key] = value
        return result

    value = json.loads(data.decode("utf-8"), parse_constant=no_constant, object_pairs_hook=unique_keys)
    if not isinstance(value, dict):
        raise ValueError("记录的顶层必须是 JSON 对象。")
    return value


def read_record(path: str | Path) -> dict:
    return deserialize(Path(path).read_bytes())


def compare(expected, actual, path: str = "$", tolerance: float = 1e-12):
    if isinstance(expected, bool) or isinstance(actual, bool):
        if type(expected) is not type(actual) or expected != actual:
            raise ValueError(f"复算不一致：{path}")
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if (not math.isfinite(expected) or not math.isfinite(actual)
                or not math.isclose(expected, actual, rel_tol=tolerance, abs_tol=tolerance)):
            raise ValueError(f"复算不一致：{path}，记录={expected!r}，复算={actual!r}")
    elif isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f"复算字段不一致：{path}")
        for key in expected:
            compare(expected[key], actual[key], f"{path}.{key}", tolerance)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ValueError(f"复算列表长度不一致：{path}")
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, f"{path}[{i}]", tolerance)
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError(f"复算不一致：{path}")


def rebuild_index(index: dict) -> dict:
    if not isinstance(index, dict) or index.get("schema") not in {f"evidence-index/{v}" for v in range(1, 11)}:
        raise ValueError("不支持的索引格式。")
    engine = index["identity"]["engine_version"]
    if engine not in SUPPORTED_ENGINE_VERSIONS:
        raise ValueError("不支持该索引的引擎版本。")
    legacy = engine == LEGACY_ENGINE_VERSION
    if index["schema"] != f"evidence-index/{schema_version(engine)}":
        raise ValueError("索引格式与引擎版本不一致。")
    source = index["source"]
    data = base64.b64decode(source["bytes_base64"], validate=True)
    dictionary = index["dictionary"]
    person_engines = {ENGINE_VERSION, DISCOURSE_ENGINE_VERSION, WINDOW_ENGINE_VERSION, ANSWER_ENGINE_VERSION, EXPANSION_ENGINE_VERSION, DAG_ENGINE_VERSION, PERSON_ENGINE_VERSION}
    dag_engines = {ENGINE_VERSION, DISCOURSE_ENGINE_VERSION, WINDOW_ENGINE_VERSION, ANSWER_ENGINE_VERSION, EXPANSION_ENGINE_VERSION, DAG_ENGINE_VERSION}
    expansion = index.get("query_expansion", {})
    rebuilt = build_index(data, source["filename"], source["encoding"],
                          dictionary["words"] if legacy else dictionary["base_words"],
                          IngestConfig(**index["identity"]["ingest_config"]),
                          book_words=None if legacy else dictionary["book_words"],
                          alias_groups=index["aliases"]["manual_groups"] if engine in person_engines else index["aliases"]["groups"] if engine == ALIAS_ENGINE_VERSION else None,
                          person_config=index["name_discovery"]["config"] if engine in person_engines else None,
                          lexicon=index["segmentation"]["lexicon"] if engine in dag_engines else None,
                          segmentation_config=index["segmentation"]["config"] if engine in dag_engines else None,
                          expansion_resource=expansion.get("resource"), expansion_config=expansion.get("config"),
                          expression_groups=expansion.get("manual_groups"), blocked_expressions=expansion.get("blocked"),
                          qa_snapshot=index.get("answering"),
                          context_config=index.get("context", {}).get("config"),
                          _engine_version=engine)
    compare(index, rebuilt, "$.index")
    return rebuilt


def replay(trace: dict) -> dict:
    if not isinstance(trace, dict) or trace.get("schema") not in {f"evidence-trace/{v}" for v in range(1, 11)}:
        raise ValueError("不支持的复算记录格式。")
    if trace.get("engine_version") not in SUPPORTED_ENGINE_VERSIONS:
        raise ValueError("不支持该记录的引擎版本。")
    legacy = trace["engine_version"] == LEGACY_ENGINE_VERSION
    if trace["schema"] != f"evidence-trace/{schema_version(trace['engine_version'])}":
        raise ValueError("记录格式与引擎版本不一致。")
    payload = {k: v for k, v in trace.items() if k != "integrity_sha256"}
    if digest(payload) != trace.get("integrity_sha256"):
        raise ValueError("记录完整性校验失败：文件内容已变化。")
    index = rebuild_index(trace["index"])
    if legacy and index["identity"]["engine_version"] != LEGACY_ENGINE_VERSION:
        raise ValueError("旧记录的索引版本不一致。")
    previous = trace["retrieval"]
    recalculated = search(index, previous["query"], BM25Config(**previous["config"]),
                          _engine_version=trace["engine_version"])
    compare(previous, recalculated, "$.retrieval")
    report = {"verified": True, "index_id": index["index_id"],
            "candidate_count": recalculated["candidate_count"],
            "top_ids": recalculated["top_ids"],
            "scores": [p["score"] for p in recalculated["candidates"][:5]],
            "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12,
            "query_policy": "base-longest/legacy-1.0" if legacy else recalculated["query_tokenization"]["policy"],
            "checks": ["原始字节与文本版本", "清洗与原文位置映射", "切分与分词轨迹",
                       "旧版查询分词规则" if legacy else "书内词频与查询选择过程",
                       *(["姓名发现规则、候选判断、原文证据与歧义", "整词保护、关系来源与去重扩展"] if trace["engine_version"] in {ENGINE_VERSION, DISCOURSE_ENGINE_VERSION, WINDOW_ENGINE_VERSION, ANSWER_ENGINE_VERSION, EXPANSION_ENGINE_VERSION, DAG_ENGINE_VERSION, PERSON_ENGINE_VERSION} else []),
                       *(["通用词典快照、初分词频、DAG 候选及实际最优路径"] if trace["engine_version"] in {ENGINE_VERSION, DISCOURSE_ENGINE_VERSION, WINDOW_ENGINE_VERSION, ANSWER_ENGINE_VERSION, EXPANSION_ENGINE_VERSION, DAG_ENGINE_VERSION} else []),
                       *(["中文 WordNet 快照、通用与本书校正", "完整词项短语区间、候选筛选、权重及同组最大贡献"] if trace["engine_version"] in {ENGINE_VERSION, DISCOURSE_ENGINE_VERSION, WINDOW_ENGINE_VERSION, ANSWER_ENGINE_VERSION, EXPANSION_ENGINE_VERSION} else []),
                       *(["问题提炼、疑问表达过滤与事实约束", "句子边界、原文区间、全部句子候选与贡献", "抽取规则、字段来源、冲突/拒答与模板答案"] if trace["engine_version"] in {ENGINE_VERSION, DISCOURSE_ENGINE_VERSION, WINDOW_ENGINE_VERSION, ANSWER_ENGINE_VERSION} else []),
                       *(["相邻句窗口、独立TF/DF、BM25贡献及排名", "证据组去重、上下文用途、两句对象绑定与字段出处"] if trace["engine_version"] in {ENGINE_VERSION, DISCOURSE_ENGINE_VERSION, WINDOW_ENGINE_VERSION} else []),
                       *(["原文明示因果方向、方法角色与逐项拒答", "原因/方法字段、事实与操作说明状态及模板"] if trace["engine_version"] in {ENGINE_VERSION, DISCOURSE_ENGINE_VERSION} else []),
                       *(["通用待答字段、身份/属性/事件角色和观点归属", "通用字段区间、原句答案与参考回答的实际内容"] if trace["engine_version"] == ENGINE_VERSION else []),
                       "词频、文档频率与段长", "全部候选及词项分数", "最终排名与并列规则"]}
    if trace["engine_version"] in {ENGINE_VERSION, DISCOURSE_ENGINE_VERSION, WINDOW_ENGINE_VERSION}:
        report.update(window_candidate_count=recalculated["window_retrieval"]["candidate_count"],
                      window_top_ids=recalculated["window_retrieval"]["top_ids"],
                      evidence_group_ids=recalculated["answer"]["evidence_group_ids"],
                      answer_status=recalculated["answer"]["status"])
    return report
