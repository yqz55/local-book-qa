"""Rebuild the frozen 1.6 pipeline using the active text's identical resources."""

import base64

from . import ANSWER_ENGINE_VERSION
from .ingest import IngestConfig
from .retrieval import build_index


def build_previous_index(index):
    """Return a real historical index, without modifying the active index.

    Both pipelines use the same source bytes, segmentation resources and book
    corrections. The old answer rules are independently instantiated by the
    frozen implementation, so new window policies cannot leak into its output.
    """
    source, dictionary = index["source"], index["dictionary"]
    segmentation, expansion = index["segmentation"], index["query_expansion"]
    return build_index(
        base64.b64decode(source["bytes_base64"], validate=True), source["filename"], source["encoding"],
        dictionary["base_words"], IngestConfig(**index["identity"]["ingest_config"]),
        book_words=dictionary["book_words"], alias_groups=index["aliases"]["manual_groups"],
        person_config=index["name_discovery"]["config"], lexicon=segmentation["lexicon"],
        segmentation_config=segmentation["config"], expansion_resource=expansion["resource"],
        expansion_config=expansion["config"], expression_groups=expansion["manual_groups"],
        blocked_expressions=expansion["blocked"], _engine_version=ANSWER_ENGINE_VERSION)


def comparison_manifest(current_index, current_result, previous_index, previous_result):
    checks = {
        "source_bytes": current_index["identity"]["bytes_sha256"] == previous_index["identity"]["bytes_sha256"],
        "source_text": current_index["source"]["text"] == previous_index["source"]["text"],
        "dictionary": current_index["dictionary"] == previous_index["dictionary"],
        "segmentation": current_index["segmentation"] == previous_index["segmentation"],
        "person_names": current_index["name_discovery"] == previous_index["name_discovery"],
        "aliases": current_index["aliases"] == previous_index["aliases"],
        "query_expansion": current_index["query_expansion"] == previous_index["query_expansion"],
        "question": current_result["query"] == previous_result["query"],
        "bm25_parameters": current_result["config"] == previous_result["config"],
    }
    if not all(checks.values()):
        failed = "、".join(name for name, passed in checks.items() if not passed)
        raise ValueError("两版对比输入不一致：" + failed)
    return {"policy": "same-source-version-comparison/1", "checks": checks,
            "query": current_result["query"], "config": current_result["config"],
            "source_sha256": current_index["identity"]["bytes_sha256"],
            "current_engine": current_index["identity"]["engine_version"],
            "previous_engine": previous_index["identity"]["engine_version"],
            "current_index_id": current_index["index_id"], "previous_index_id": previous_index["index_id"]}
