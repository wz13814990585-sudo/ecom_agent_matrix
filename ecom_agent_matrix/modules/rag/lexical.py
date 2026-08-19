"""Independent bounded lexical retrieval channel。"""
from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

from ecom_agent_matrix.config.constants import TABLE_VECTOR_GOODS
from ecom_agent_matrix.db.base import AsyncPGClient
from ecom_agent_matrix.modules.rag.lexicon import tokenize


def _query_tokens(query: str, limit: int = 8) -> list[str]:
    unique: list[str] = []
    for token in tokenize(query):
        cleaned = str(token).strip()
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
        if len(unique) >= limit:
            break
    return unique


async def lexical_search(
    query: str,
    lang: str,
    price_max: float | None,
    candidate_k: int,
) -> list[dict[str, Any]]:
    """Use parameterized ILIKE to bound candidates, then BM25-rank only that set."""
    bounded_k = min(max(int(candidate_k), 1), 80)
    tokens = _query_tokens(query)
    if not tokens:
        return []
    patterns = [f"%{token}%" for token in tokens]
    match_expr = " + ".join("CASE WHEN chunk_text ILIKE %s THEN 1 ELSE 0 END" for _ in tokens)
    where_expr = " OR ".join("chunk_text ILIKE %s" for _ in tokens)
    sql = f"""
    SELECT goods_sku, chunk_text, meta_json, ({match_expr}) AS match_count
    FROM {TABLE_VECTOR_GOODS}
    WHERE (%s = '' OR lang = %s)
      AND ({where_expr})
    """
    params: list[Any] = [*patterns, lang or "", lang or "", *patterns]
    if price_max is not None:
        sql += " AND (meta_json->>'price')::float <= %s"
        params.append(price_max)
    sql += " ORDER BY match_count DESC LIMIT %s"
    params.append(bounded_k)
    rows = await AsyncPGClient.execute_sql(sql, params)
    if not rows:
        return []

    candidates = [
        {
            "sku": row[0],
            "chunk_text": row[1],
            "meta": row[2],
            "lexical_match_count": int(row[3] or 0),
        }
        for row in rows
    ]
    corpus = [tokenize(item["chunk_text"]) or ["_"] for item in candidates]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokens or ["_"])
    ranked: list[dict[str, Any]] = []
    for item, score in zip(candidates, scores):
        ranked.append({**item, "bm25_score": float(score)})
    return sorted(
        ranked,
        key=lambda item: (item["bm25_score"], item["lexical_match_count"]),
        reverse=True,
    )[:bounded_k]


__all__ = ["lexical_search"]
