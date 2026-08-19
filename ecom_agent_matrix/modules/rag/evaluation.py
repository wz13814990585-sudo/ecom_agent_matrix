"""Deterministic, retrieval-only RAG evaluation contract."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RAGEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1)
    relevant_source_ids: list[str] = Field(min_length=1)

    @field_validator("relevant_source_ids")
    @classmethod
    def unique_non_blank_ids(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        if not normalized:
            raise ValueError("relevant_source_ids must contain a non-blank source id")
        return normalized


class RAGEvalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int = Field(ge=1)
    case_count: int = Field(ge=0)
    hit_rate_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    mrr_at_k: float = Field(ge=0, le=1)
    ndcg_at_k: float = Field(ge=0, le=1)


def _source_id(item: object) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        return str(item.get("source_id") or "")
    return str(getattr(item, "source_id", "") or "")


def evaluate_ranked_results(
    cases: Sequence[RAGEvalCase],
    ranked_results: Mapping[str, Sequence[object]],
    *,
    k: int,
) -> RAGEvalMetrics:
    """Average binary-relevance HitRate, Recall, MRR and nDCG at K."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not cases:
        return RAGEvalMetrics(
            k=k,
            case_count=0,
            hit_rate_at_k=0,
            recall_at_k=0,
            mrr_at_k=0,
            ndcg_at_k=0,
        )

    hits = recalls = reciprocal_ranks = ndcgs = 0.0
    for case in cases:
        relevant = set(case.relevant_source_ids)
        ranked_ids = [_source_id(item) for item in ranked_results.get(case.query, ())[:k]]
        relevant_ranks = [
            rank for rank, source_id in enumerate(ranked_ids, start=1) if source_id in relevant
        ]
        if relevant_ranks:
            hits += 1.0
            reciprocal_ranks += 1.0 / relevant_ranks[0]
        recalls += len(set(ranked_ids) & relevant) / len(relevant)
        dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
        ideal_count = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcgs += dcg / idcg if idcg else 0.0

    count = len(cases)
    return RAGEvalMetrics(
        k=k,
        case_count=count,
        hit_rate_at_k=hits / count,
        recall_at_k=recalls / count,
        mrr_at_k=reciprocal_ranks / count,
        ndcg_at_k=ndcgs / count,
    )


__all__ = ["RAGEvalCase", "RAGEvalMetrics", "evaluate_ranked_results"]
