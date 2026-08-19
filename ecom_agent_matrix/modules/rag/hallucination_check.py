"""幻觉抑制：同义词扩展 + 停用词 + Cross-Encoder 语义分 + 动态阈值。"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.modules.rag.lexicon import expand_synonyms, tokenize

KEYWORD_WEIGHT = 0.35
SEMANTIC_WEIGHT = 0.65


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """懒加载轻量 Cross-Encoder。优先本地路径，避免首次强制下载大模型。"""
    from pathlib import Path

    from sentence_transformers import CrossEncoder

    local = Path(settings.RERANK_MODEL_PATH)
    if local.exists():
        return CrossEncoder(str(local))

    hf_name = getattr(settings, "RERANK_MODEL_HF", "BAAI/bge-reranker-base")
    # 仅当显式允许下载时才走 HuggingFace
    import os

    if os.getenv("ECOM_DOWNLOAD_RERANKER", "").lower() in {"1", "true", "yes"}:
        return CrossEncoder(hf_name)
    raise FileNotFoundError(
        f"未找到本地 Cross-Encoder：{local}。将仅使用关键词/同义词打分。"
        "如需语义打分，请下载 bge-reranker-base 到该路径，"
        "或设置环境变量 ECOM_DOWNLOAD_RERANKER=1"
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def keyword_overlap_score(query: str, chunk: str) -> float:
    """
    改进后的关键词重合度：
    1) 中英分词，而不是按空格切
    2) 去掉停用词，避免「的 / 这款」稀释分母
    3) 同义词扩展后再求交，使「半袖」能命中「短袖」
    分数 = |query 扩展词 ∩ chunk 扩展词| / |query 扩展词|
    """
    query_tokens = tokenize(query)
    chunk_tokens = tokenize(chunk)
    if not query_tokens:
        return 0.0
    query_set = expand_synonyms(query_tokens)
    chunk_set = expand_synonyms(chunk_tokens)
    hit = len(query_set & chunk_set)
    return hit / len(query_set)


def semantic_score(query: str, chunk: str) -> Optional[float]:
    """Cross-Encoder 相关性，映射到 0~1；模型不可用时返回 None。"""
    try:
        model = _get_cross_encoder()
        raw = float(model.predict([(query, chunk)], show_progress_bar=False)[0])
        return _sigmoid(raw)
    except Exception:
        return None


def text_relevance_score(query: str, chunk: str) -> float:
    """综合相关性：关键词重合 + Cross-BERT 语义分。"""
    kw = keyword_overlap_score(query, chunk)
    sem = semantic_score(query, chunk)
    if sem is None:
        return kw
    return KEYWORD_WEIGHT * kw + SEMANTIC_WEIGHT * sem


def dynamic_threshold(query: str) -> float:
    """
    短 query 调高阈值（词少、误召回代价大），
    长 query 降低阈值（词多、单词语重合被稀释）。
    """
    n = len(tokenize(query))
    if n <= 2:
        return 0.40
    if n <= 5:
        return 0.25
    return 0.15


def filter_irrelevant_chunks(
    query: str,
    chunk_list: list[dict],
    threshold: Optional[float] = None,
) -> list[dict]:
    """过滤低相关文档；threshold 为空时按 query 长度自动设定。"""
    cut = dynamic_threshold(query) if threshold is None else threshold
    valid_chunks: list[dict] = []
    for item in chunk_list:
        text = item["chunk_text"]
        kw = keyword_overlap_score(query, text)
        sem = semantic_score(query, text)
        score = kw if sem is None else KEYWORD_WEIGHT * kw + SEMANTIC_WEIGHT * sem
        if score >= cut:
            enriched = dict(item)
            enriched["relevance_score"] = round(score, 4)
            enriched["keyword_score"] = round(kw, 4)
            enriched["semantic_score"] = None if sem is None else round(sem, 4)
            enriched["threshold"] = cut
            valid_chunks.append(enriched)
    return valid_chunks
