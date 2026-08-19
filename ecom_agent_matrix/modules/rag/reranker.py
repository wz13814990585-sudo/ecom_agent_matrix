"""Batch CrossEncoder reranking with deterministic keyword fallback。"""
from __future__ import annotations

import asyncio
import math
from typing import Any

from ecom_agent_matrix.modules.rag.hallucination_check import (
    KEYWORD_WEIGHT,
    SEMANTIC_WEIGHT,
    _get_cross_encoder,
    dynamic_threshold,
    keyword_overlap_score,
)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


async def rerank_documents_detailed(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    *,
    threshold: float | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if not candidates:
        return [], "none"
    keyword_scores = [
        keyword_overlap_score(query, str(candidate.get("chunk_text") or ""))
        for candidate in candidates
    ]
    semantic_scores: list[float] | None = None
    mode = "cross_encoder"
    try:
        model = _get_cross_encoder()
        pairs = [
            (query, str(candidate.get("chunk_text") or ""))
            for candidate in candidates
        ]
        raw_scores = await asyncio.to_thread(
            model.predict,
            pairs,
            show_progress_bar=False,
        )
        semantic_scores = [_sigmoid(float(score)) for score in raw_scores]
        if len(semantic_scores) != len(candidates):
            raise ValueError("CrossEncoder returned an invalid score count")
    except Exception:
        mode = "keyword_fallback"

    enriched: list[dict[str, Any]] = []
    cut = dynamic_threshold(query) if threshold is None else float(threshold)
    for index, candidate in enumerate(candidates):
        keyword_score = keyword_scores[index]
        semantic_score = semantic_scores[index] if semantic_scores is not None else None
        if semantic_score is None:
            rrf_component = min(max(float(candidate.get("rrf_score") or 0) * 60, 0), 1)
            relevance = 0.8 * keyword_score + 0.2 * rrf_component
        else:
            relevance = KEYWORD_WEIGHT * keyword_score + SEMANTIC_WEIGHT * semantic_score
        if relevance < cut:
            continue
        enriched.append(
            {
                **dict(candidate),
                "keyword_score": round(keyword_score, 6),
                "semantic_score": (
                    None if semantic_score is None else round(semantic_score, 6)
                ),
                "relevance_score": round(relevance, 6),
                "threshold": cut,
                "rerank_mode": mode,
            }
        )
    enriched.sort(key=lambda item: float(item["relevance_score"]), reverse=True)
    return enriched[: max(0, int(top_k))], mode


async def rerank_documents(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    *,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    documents, _mode = await rerank_documents_detailed(
        query,
        candidates,
        top_k,
        threshold=threshold,
    )
    return documents


__all__ = ["rerank_documents", "rerank_documents_detailed"]
