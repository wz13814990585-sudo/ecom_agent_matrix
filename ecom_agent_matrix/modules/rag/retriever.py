"""混合检索（BM25 + 向量 RRF），含 Redis 结果缓存。"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from ecom_agent_matrix.config.constants import TABLE_VECTOR_GOODS
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.db.base import AsyncPGClient
from ecom_agent_matrix.db.redis_client import AsyncRedisClient
from ecom_agent_matrix.modules.rag.embedding import get_text_embedding
from ecom_agent_matrix.modules.rag.hallucination_check import filter_irrelevant_chunks
from ecom_agent_matrix.modules.rag.rate_limiter import get_rag_semaphore

logger = setup_logger("rag.retriever")

RRF_K = 60


def _cache_key(query: str, lang: str, price_max: Optional[float], top_k: int) -> str:
    payload = json.dumps(
        {
            "index_version": settings.RAG_INDEX_VERSION,
            "retrieval_version": settings.RAG_RETRIEVAL_VERSION,
            "q": query.strip().lower(),
            "lang": lang,
            "price_max": price_max,
            "top_k": top_k,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (
        f"rag:retrieve:{settings.RAG_INDEX_VERSION}:"
        f"{settings.RAG_RETRIEVAL_VERSION}:{digest}"
    )


async def _load_cache(key: str) -> Optional[list[dict]]:
    if not settings.RAG_CACHE_ENABLED:
        return None
    try:
        redis = await AsyncRedisClient.get_client()
        raw = await redis.get(key)
        if not raw:
            return None
        decoded = json.loads(raw)
        return decoded if isinstance(decoded, list) else None
    except json.JSONDecodeError as exc:
        logger.warning(
            "rag_cache_decode_failed",
            extra={"event": "rag_cache_decode_failed", "error_type": type(exc).__name__},
        )
        return None
    except Exception as exc:
        logger.warning(
            "rag_cache_read_failed",
            extra={"event": "rag_cache_read_failed", "error_type": type(exc).__name__},
        )
        return None


async def _save_cache(key: str, docs: list[dict]) -> None:
    if not settings.RAG_CACHE_ENABLED:
        return
    try:
        redis = await AsyncRedisClient.get_client()
        await redis.set(key, json.dumps(docs, ensure_ascii=False), ex=settings.RAG_CACHE_TTL)
    except Exception as exc:
        logger.warning(
            "rag_cache_write_failed",
            extra={"event": "rag_cache_write_failed", "error_type": type(exc).__name__},
        )


async def vector_search(
    query_vec: list[float], lang: str, price_max: float = None, top_k: int = 10
) -> List[Dict]:
    """PGVector 向量相似度检索，支持价格筛选。"""
    # aiopg/psycopg2 需要向量字面量字符串，而非 Python list
    if isinstance(query_vec, list):
        vec_lit = "[" + ",".join(f"{float(x):.6f}" for x in query_vec) + "]"
    else:
        vec_lit = query_vec

    base_sql = f"""
    SELECT goods_sku, chunk_text, meta_json, embedding <-> %s::vector AS dist
    FROM {TABLE_VECTOR_GOODS}
    WHERE (%s = '' OR lang = %s)
    """
    params: list = [vec_lit, lang or "", lang or ""]
    if price_max is not None:
        base_sql += " AND (meta_json->>'price')::float <= %s"
        params.append(price_max)
    base_sql += f" ORDER BY dist ASC LIMIT {top_k};"
    res = await AsyncPGClient.execute_sql(base_sql, params)
    return [
        {
            "sku": row[0],
            "chunk_text": row[1],
            "meta": row[2],
            "score": 1.0 / (1.0 + float(row[3])),
        }
        for row in res
    ]


def build_bm25_corpus(all_chunks: List[Dict]) -> tuple[BM25Okapi, List[Dict]]:
    # 中文按空格切会失效，统一用 jieba/lexicon 分词
    from ecom_agent_matrix.modules.rag.lexicon import tokenize

    texts = [tokenize(item["chunk_text"]) for item in all_chunks]
    # BM25Okapi 需要非空 token 列表
    texts = [t if t else ["_"] for t in texts]
    return BM25Okapi(texts), all_chunks


def rrf_fuse(vec_list: List[Dict], bm25_list: List[Dict]) -> List[Dict]:
    rank_map: dict[str, float] = {}
    for rank, item in enumerate(vec_list):
        key = item["chunk_text"]
        rank_map[key] = rank_map.get(key, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, item in enumerate(bm25_list):
        key = item["chunk_text"]
        rank_map[key] = rank_map.get(key, 0) + 1.0 / (RRF_K + rank + 1)

    unique_items: dict[str, dict] = {}
    for item in vec_list + bm25_list:
        unique_items[item["chunk_text"]] = item
    for text in unique_items:
        unique_items[text]["rrf_score"] = rank_map[text]
    return sorted(unique_items.values(), key=lambda x: x["rrf_score"], reverse=True)


async def _hybrid_retrieve_uncached(
    query: str, lang: str, price_max: float = None, top_k: int = 8
) -> List[Dict]:
    from ecom_agent_matrix.modules.rag.lexicon import tokenize

    query_vec = await get_text_embedding(query)
    vec_res = await vector_search(query_vec, lang, price_max, top_k * 2)
    # 指定语种无结果时，放宽到全语种再搜一轮
    if not vec_res and lang:
        vec_res = await vector_search(query_vec, "", price_max, top_k * 2)
    if not vec_res:
        return []

    bm25_model, corpus = build_bm25_corpus(vec_res)
    token_query = tokenize(query) or query.split() or ["_"]
    bm25_scores = bm25_model.get_scores(token_query)
    bm25_res = []
    for idx, score in enumerate(bm25_scores):
        item = dict(corpus[idx])
        item["bm25_score"] = float(score)
        bm25_res.append(item)

    fuse_result = rrf_fuse(vec_res, bm25_res)
    valid_result = filter_irrelevant_chunks(query, fuse_result)
    return valid_result[:top_k]


async def hybrid_retrieve(
    query: str,
    lang: str,
    price_max: float = None,
    top_k: int = 8,
    *,
    task_id: str = "",
) -> tuple[list[dict], bool, float]:
    """
    混合检索统一入口。
    返回: (文档列表, 是否命中缓存, 耗时毫秒)
    """
    start = time.perf_counter()
    cache_key = _cache_key(query, lang, price_max, top_k)

    cached_docs = await _load_cache(cache_key)
    if cached_docs is not None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "rag_cache_hit",
            extra={
                "event": "rag_cache_hit",
                "task_id": task_id,
                "query": query,
                "lang": lang,
                "recall_count": len(cached_docs),
                "latency_ms": round(elapsed_ms, 2),
                "cached": True,
            },
        )
        return cached_docs, True, elapsed_ms

    sem = get_rag_semaphore()
    async with sem:
        docs = await _hybrid_retrieve_uncached(query, lang, price_max, top_k)

    await _save_cache(cache_key, docs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "rag_retrieve_done",
        extra={
            "event": "rag_retrieve_done",
            "task_id": task_id,
            "query": query,
            "lang": lang,
            "recall_count": len(docs),
            "latency_ms": round(elapsed_ms, 2),
            "cached": False,
        },
    )
    return docs, False, elapsed_ms
