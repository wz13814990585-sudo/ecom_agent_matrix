"""RAG service typed contracts。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RAGRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1)
    lang: str = "en"
    top_k: int = Field(default=8, ge=1, le=20)
    price_max: float | None = Field(default=None, ge=0)
    task_id: str = ""

    @field_validator("query")
    @classmethod
    def non_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class RAGDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    source_id: str
    sku: str | None = None
    title: str = ""
    chunk_text: str
    lang: str | None = None
    score: float | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    relevance_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    source_id: str
    title: str = ""
    sku: str | None = None
    snippet: str = ""


class RAGRetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    documents: list[RAGDocument] = Field(default_factory=list)
    citations: list[RAGCitation] = Field(default_factory=list)
    cached: bool = False
    recall_count: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    retrieval_version: str
    retrieval_mode: Literal["hybrid", "vector_only", "lexical_only", "none"] = "none"
    degraded: bool = False
    channel_errors: dict[str, str] = Field(default_factory=dict)
    candidate_counts: dict[str, int] = Field(default_factory=dict)
    diagnostics: dict[str, float | str] = Field(default_factory=dict)
    error_code: str = ""
    error_msg: str = ""


class RAGAnswerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    answer: str
    documents: list[RAGDocument] = Field(default_factory=list)
    citations: list[RAGCitation] = Field(default_factory=list)
    grounded: bool
    answer_source: str
    cached: bool
    retrieval_latency_ms: float = Field(default=0, ge=0)
    total_latency_ms: float = Field(default=0, ge=0)
    invalid_citation_ids: list[str] = Field(default_factory=list)
    citation_status: Literal["valid", "missing", "invalid", "none"] = "none"
    retrieval_mode: Literal["hybrid", "vector_only", "lexical_only", "none"] = "none"
    degraded: bool = False
    channel_errors: dict[str, str] = Field(default_factory=dict)
    candidate_counts: dict[str, int] = Field(default_factory=dict)
    error_code: str = ""
    error_msg: str = ""


class HybridRetrievalResult(BaseModel):
    """Retriever internal detailed result; raw documents have no citation rank yet."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    raw_documents: list[dict[str, Any]] = Field(default_factory=list)
    mode: Literal["hybrid", "vector_only", "lexical_only", "none"]
    degraded: bool = False
    channel_errors: dict[str, str] = Field(default_factory=dict)
    candidate_counts: dict[str, int] = Field(default_factory=dict)
    diagnostics: dict[str, float | str] = Field(default_factory=dict)
    cached: bool = False
    latency_ms: float = Field(default=0, ge=0)
    error_code: str = ""
