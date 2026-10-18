"""Frozen 1.2 resources. Only replay and one-time migration use these examples."""

from .tokenization import PROTECTED_RULES

LEGACY_ALIAS_GROUPS = [["林黛玉", "黛玉"]]
ALIAS_RULES = {
    **PROTECTED_RULES,
    "A002": "问题中的完整别名词按本书显式别名组双向展开一层；原词与其他组成员均参与检索，每个去重词权重为 1。",
    "A003": "正文仅按实际出现的、不重叠的完整词计 TF；不把林黛玉中的黛玉重复计数。别名扩展词没有虚构的问题原文位置。",
    "A004": "同一名称只能属于一个别名组；不按单字、子串、共同后缀或传递关系猜测其他实体。",
}
