"""Deterministic knowledge retrieval policy。"""
from __future__ import annotations

import re

_KNOWLEDGE_HINT = re.compile(
    r"(材质|规格|怎么用|如何使用|介绍|知识|防水|尺寸|清洗|保养|面料|成分|"
    r"背包|商品|款式|退款规则|退货政策|店铺规则|faq|how to|what is|"
    r"material|care|wash|fabric|refund\s+policy|return\s+policy)",
    re.IGNORECASE,
)


def should_retrieve_knowledge(query: str, explicit: bool | None) -> bool:
    if explicit is False:
        return False
    if explicit is True:
        return True
    return bool(_KNOWLEDGE_HINT.search(query or ""))
