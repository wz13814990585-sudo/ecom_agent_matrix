"""混合检索（BM25 + 向量 RRF），含 Redis 结果缓存。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional

from ecom_agent_matrix.config.constants import TABLE_VECTOR_GOODS
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.db.base import AsyncPGClient
from ecom_agent_matrix.db.redis_client import AsyncRedisClient
from ecom_agent_matrix.modules.rag.embedding import get_text_embedding
from ecom_agent_matrix.modules.rag.formatter import stable_source_id
from ecom_agent_matrix.modules.rag.lexical import lexical_search
from ecom_agent_matrix.modules.rag.rate_limiter import get_rag_semaphore
from ecom_agent_matrix.modules.rag.reranker import rerank_documents_detailed
from ecom_agent_matrix.modules.rag.schemas import HybridRetrievalResult

logger = setup_logger("rag.retriever")

RRF_K = 60


class EmbeddingChannelError(RuntimeError):
    pass


def candidate_limit(top_k: int) -> int:
    return min(max(int(top_k) * 4, 20), 80)


def _query_log_fields(query: str) -> dict[str, int | str]:
    digest = hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:16]
    return {"query_hash": digest, "query_length": len(query or "")}


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
    bounded_k = min(max(int(top_k), 1), 80)
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
    base_sql += " ORDER BY dist ASC LIMIT %s;"
    params.append(bounded_k)
    res = await AsyncPGClient.execute_sql(base_sql, params)
    return [
        {
            "sku": row[0],
            "chunk_text": row[1],
            "meta": row[2],
            "score": 1.0 / (1.0 + float(row[3])),
            "vector_score": 1.0 / (1.0 + float(row[3])),
        }
        for row in res
    ]


def rrf_fuse(vec_list: List[Dict], bm25_list: List[Dict]) -> List[Dict]:
    rank_map: dict[str, float] = {}
    for rank, item in enumerate(vec_list):
        key = stable_source_id(item)
        rank_map[key] = rank_map.get(key, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, item in enumerate(bm25_list):
        key = stable_source_id(item)
        rank_map[key] = rank_map.get(key, 0) + 1.0 / (RRF_K + rank + 1)

    unique_items: dict[str, dict] = {}
    for item in vec_list + bm25_list:
        key = stable_source_id(item)
        unique_items[key] = {**unique_items.get(key, {}), **dict(item), "source_id": key}
    for source_id in unique_items:
        unique_items[source_id] = {
            **unique_items[source_id],
            "rrf_score": rank_map[source_id],
        }
    return sorted(unique_items.values(), key=lambda x: x["rrf_score"], reverse=True)


async def _vector_channel(
    query: str,
    lang: str,
    price_max: float | None,
    candidate_k: int,
) -> tuple[list[dict], float]:
    started = time.perf_counter()
    try:
        query_vec = await get_text_embedding(query)
    except Exception as exc:
        raise EmbeddingChannelError("embedding unavailable") from exc
    documents = await vector_search(query_vec, lang, price_max, candidate_k)
    if not documents and lang:
        documents = await vector_search(query_vec, "", price_max, candidate_k)
    return documents, (time.perf_counter() - started) * 1000


async def _lexical_channel(
    query: str,
    lang: str,
    price_max: float | None,
    candidate_k: int,
) -> tuple[list[dict], float]:
    started = time.perf_counter()
    documents = await lexical_search(query, lang, price_max, candidate_k)
    if not documents and lang:
        documents = await lexical_search(query, "", price_max, candidate_k)
    return documents, (time.perf_counter() - started) * 1000


async def _hybrid_retrieve_detailed_uncached(
    query: str,
    lang: str,
    price_max: float | None = None,
    top_k: int = 8,
) -> HybridRetrievalResult:
    started = time.perf_counter()
    candidate_k = candidate_limit(top_k)
    vector_outcome, lexical_outcome = await asyncio.gather(
        _vector_channel(query, lang, price_max, candidate_k),
        _lexical_channel(query, lang, price_max, candidate_k),
        return_exceptions=True,
    )
    errors: dict[str, str] = {}
    vector_docs: list[dict] = []
    lexical_docs: list[dict] = []
    vector_ms = 0.0
    lexical_ms = 0.0
    if isinstance(vector_outcome, BaseException):
        errors["vector"] = (
            "EMBEDDING_ERROR"
            if isinstance(vector_outcome, EmbeddingChannelError)
            else type(vector_outcome).__name__
        )
    else:
        vector_docs, vector_ms = vector_outcome
    if isinstance(lexical_outcome, BaseException):
        errors["lexical"] = type(lexical_outcome).__name__
    else:
        lexical_docs, lexical_ms = lexical_outcome

    vector_failed = "vector" in errors
    lexical_failed = "lexical" in errors
    if vector_failed and lexical_failed:
        return HybridRetrievalResult(
            success=False,
            mode="none",
            degraded=True,
            channel_errors=errors,
            candidate_counts={"vector": 0, "lexical": 0, "fused": 0, "reranked": 0},
            diagnostics={
                "vector_ms": vector_ms,
                "lexical_ms": lexical_ms,
                "fusion_ms": 0.0,
                "rerank_ms": 0.0,
                "total_ms": (time.perf_counter() - started) * 1000,
            },
            latency_ms=(time.perf_counter() - started) * 1000,
            error_code="RETRIEVAL_ERROR",
        )

    mode = "hybrid"
    if vector_failed:
        mode = "lexical_only"
    elif lexical_failed:
        mode = "vector_only"
    fusion_started = time.perf_counter()
    fused = rrf_fuse(vector_docs, lexical_docs)
    fusion_ms = (time.perf_counter() - fusion_started) * 1000
    rerank_started = time.perf_counter()
    reranked, rerank_mode = await rerank_documents_detailed(query, fused, top_k)
    rerank_ms = (time.perf_counter() - rerank_started) * 1000
    total_ms = (time.perf_counter() - started) * 1000
    return HybridRetrievalResult(
        success=True,
        raw_documents=reranked,
        mode=mode,
        degraded=bool(errors),
        channel_errors=errors,
        candidate_counts={
            "vector": len(vector_docs),
            "lexical": len(lexical_docs),
            "fused": len(fused),
            "reranked": len(reranked),
        },
        diagnostics={
            "vector_ms": vector_ms,
            "lexical_ms": lexical_ms,
            "fusion_ms": fusion_ms,
            "rerank_ms": rerank_ms,
            "total_ms": total_ms,
            "rerank_mode": rerank_mode,
        },
        latency_ms=total_ms,
    )


async def _hybrid_retrieve_uncached(
    query: str, lang: str, price_max: float = None, top_k: int = 8
) -> List[Dict]:
    result = await _hybrid_retrieve_detailed_uncached(query, lang, price_max, top_k)
    if not result.success:
        raise RuntimeError("all retrieval channels failed")
    return result.raw_documents


async def hybrid_retrieve_detailed(
    query: str,
    lang: str,
    price_max: float | None = None,
    top_k: int = 8,
    *,
    task_id: str = "",
) -> HybridRetrievalResult:
    started = time.perf_counter()
    cache_key = _cache_key(query, lang, price_max, top_k)
    cached_docs = await _load_cache(cache_key)
    if cached_docs is not None:
        elapsed = (time.perf_counter() - started) * 1000
        return HybridRetrievalResult(
            success=True,
            raw_documents=cached_docs,
            mode="hybrid",
            cached=True,
            candidate_counts={
                "vector": 0,
                "lexical": 0,
                "fused": len(cached_docs),
                "reranked": len(cached_docs),
            },
            diagnostics={"total_ms": elapsed, "rerank_mode": "cached"},
            latency_ms=elapsed,
        )
    sem = get_rag_semaphore()
    async with sem:
        result = await _hybrid_retrieve_detailed_uncached(query, lang, price_max, top_k)
    # A degraded result is usable for this request but its channel mode cannot be
    # reconstructed from the legacy list-only cache payload.
    if result.success and not result.degraded:
        await _save_cache(cache_key, result.raw_documents)
    logger.info(
        "rag_retrieve_done",
        extra={
            "event": "rag_retrieve_done",
            "task_id": task_id,
            **_query_log_fields(query),
            "lang": lang,
            "recall_count": len(result.raw_documents),
            "latency_ms": round(result.latency_ms, 2),
            "cached": False,
            "retrieval_mode": result.mode,
            "degraded": result.degraded,
        },
    )
    return result


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
                **_query_log_fields(query),
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
            **_query_log_fields(query),
            "lang": lang,
            "recall_count": len(docs),
            "latency_ms": round(elapsed_ms, 2),
            "cached": False,
        },
    )
    return docs, False, elapsed_ms
