"""MCP-independent typed RAG retrieval and answer service。"""
from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.modules.rag.formatter import (
    citations_for_documents,
    format_rag_context,
    normalize_documents,
    validate_answer_citations,
)
from ecom_agent_matrix.modules.rag.retriever import hybrid_retrieve_detailed
from ecom_agent_matrix.modules.rag.schemas import (
    RAGAnswerResult,
    RAGRequest,
    RAGRetrievalResult,
)
from ecom_agent_matrix.modules.utils.llm_explain import llm_explain
from ecom_agent_matrix.core.security import TenantScope, require_tenant_scope

logger = setup_logger("rag.service")
INVALID_REQUEST = "INVALID_REQUEST"
EMBEDDING_ERROR = "EMBEDDING_ERROR"
RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
GENERATION_ERROR = "GENERATION_ERROR"

RAG_ANSWER_SYSTEM = (
    "你是跨境独立站知识库助手。只能依据提供的 [S1] 形式文档回答。"
    "事实陈述尽量附带 [S1] 形式引用；不得引用不存在的编号。"
    "文档不足时明确说明知识库暂无足够信息，不得自由编造。"
)


def _no_knowledge_answer(request: RAGRequest) -> str:
    if request.lang.startswith("zh"):
        return f"知识库暂无与「{request.query}」相关的足够信息，请补充商品名、SKU 或具体规则。"
    return (
        f'The knowledge base does not contain enough information about "{request.query}". '
        "Please provide a product name, SKU, or specific policy."
    )


class RAGService:
    async def retrieve(
        self, request: RAGRequest | dict[str, Any], *, scope: TenantScope | None = None
    ) -> RAGRetrievalResult:
        started = time.perf_counter()
        try:
            typed = request if isinstance(request, RAGRequest) else RAGRequest.model_validate(request)
        except (ValidationError, TypeError, ValueError):
            return RAGRetrievalResult(
                success=False,
                retrieval_version=settings.RAG_RETRIEVAL_VERSION,
                error_code=INVALID_REQUEST,
                error_msg="Invalid RAG request",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        try:
            require_tenant_scope(
                scope or TenantScope(),
                production=str(settings.APP_ENV).lower() == "production",
            )
            detailed = await hybrid_retrieve_detailed(
                typed.query,
                typed.lang,
                typed.price_max,
                typed.top_k,
                task_id=typed.task_id,
                scope=scope,
            )
            if not detailed.success:
                return RAGRetrievalResult(
                    success=False,
                    retrieval_version=settings.RAG_RETRIEVAL_VERSION,
                    retrieval_mode=detailed.mode,
                    degraded=detailed.degraded,
                    channel_errors=detailed.channel_errors,
                    candidate_counts=detailed.candidate_counts,
                    diagnostics=detailed.diagnostics,
                    error_code=detailed.error_code or RETRIEVAL_ERROR,
                    error_msg="RAG retrieval failed",
                    latency_ms=detailed.latency_ms,
                )
            documents = normalize_documents(list(detailed.raw_documents or []))
            citations = citations_for_documents(documents)
            return RAGRetrievalResult(
                success=True,
                documents=documents,
                citations=citations,
                cached=detailed.cached,
                recall_count=len(documents),
                latency_ms=max(float(detailed.latency_ms), 0),
                retrieval_version=settings.RAG_RETRIEVAL_VERSION,
                retrieval_mode=detailed.mode,
                degraded=detailed.degraded,
                channel_errors=detailed.channel_errors,
                candidate_counts=detailed.candidate_counts,
                diagnostics=detailed.diagnostics,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.error(
                "rag_service_retrieval_failed",
                extra={
                    "event": "rag_service_retrieval_failed",
                    "task_id": typed.task_id,
                    "error_type": type(exc).__name__,
                    "latency_ms": latency_ms,
                },
            )
            return RAGRetrievalResult(
                success=False,
                retrieval_version=settings.RAG_RETRIEVAL_VERSION,
                error_code=RETRIEVAL_ERROR,
                error_msg="RAG retrieval failed",
                latency_ms=latency_ms,
            )

    async def answer(
        self, request: RAGRequest | dict[str, Any], *, scope: TenantScope | None = None
    ) -> RAGAnswerResult:
        started = time.perf_counter()
        try:
            typed = request if isinstance(request, RAGRequest) else RAGRequest.model_validate(request)
        except (ValidationError, TypeError, ValueError):
            return RAGAnswerResult(
                success=False,
                answer="",
                grounded=False,
                answer_source="validation",
                cached=False,
                retrieval_latency_ms=0,
                total_latency_ms=round((time.perf_counter() - started) * 1000, 2),
                error_code=INVALID_REQUEST,
                error_msg="Invalid RAG request",
            )
        retrieval = await self.retrieve(typed, scope=scope)
        if not retrieval.success:
            return RAGAnswerResult(
                success=False,
                answer="",
                grounded=False,
                answer_source="retrieval_error",
                cached=retrieval.cached,
                retrieval_latency_ms=retrieval.latency_ms,
                total_latency_ms=round((time.perf_counter() - started) * 1000, 2),
                retrieval_mode=retrieval.retrieval_mode,
                degraded=retrieval.degraded,
                channel_errors=retrieval.channel_errors,
                candidate_counts=retrieval.candidate_counts,
                error_code=retrieval.error_code,
                error_msg=retrieval.error_msg,
            )
        if not retrieval.documents:
            return RAGAnswerResult(
                success=True,
                answer=_no_knowledge_answer(typed),
                grounded=False,
                answer_source="no_knowledge",
                cached=retrieval.cached,
                retrieval_latency_ms=retrieval.latency_ms,
                total_latency_ms=round((time.perf_counter() - started) * 1000, 2),
                retrieval_mode=retrieval.retrieval_mode,
                degraded=retrieval.degraded,
                channel_errors=retrieval.channel_errors,
                candidate_counts=retrieval.candidate_counts,
            )

        context = format_rag_context(retrieval.documents)
        fallback = (
            f"根据知识库检索结果：\n{context}"
            if typed.lang.startswith("zh")
            else f"Based on the retrieved knowledge:\n{context}"
        )
        try:
            answer, source, generation_error = await llm_explain(
                system_prompt=RAG_ANSWER_SYSTEM,
                user_prompt=(
                    f"Language: {typed.lang}\nUser question: {typed.query}\n\n"
                    f"Documents:\n{context}\n\nAnswer only from these documents with citations."
                ),
                fallback=fallback,
                max_tokens=int(settings.AGENT_LLM_EXPLAIN_MAX_TOKENS),
                temperature=0.2,
            )
        except Exception as exc:
            logger.error(
                "rag_service_generation_failed",
                extra={
                    "event": "rag_service_generation_failed",
                    "task_id": typed.task_id,
                    "error_type": type(exc).__name__,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            answer, source, generation_error = fallback, "template", "generation_failed"
        clean_answer, valid, invalid = validate_answer_citations(
            answer,
            retrieval.citations,
        )
        grounded = bool(valid) and not invalid
        citation_status = "invalid" if invalid else ("valid" if valid else "missing")
        generation_failed = bool(generation_error) and generation_error not in {"no_api_key"}
        return RAGAnswerResult(
            success=True,
            answer=clean_answer,
            documents=retrieval.documents,
            citations=retrieval.citations,
            grounded=grounded,
            answer_source=source,
            cached=retrieval.cached,
            retrieval_latency_ms=retrieval.latency_ms,
            total_latency_ms=round((time.perf_counter() - started) * 1000, 2),
            invalid_citation_ids=invalid,
            citation_status=citation_status,
            retrieval_mode=retrieval.retrieval_mode,
            degraded=retrieval.degraded,
            channel_errors=retrieval.channel_errors,
            candidate_counts=retrieval.candidate_counts,
            error_code=GENERATION_ERROR if generation_failed else "",
            error_msg="RAG answer generation used safe fallback" if generation_failed else "",
        )


rag_service = RAGService()

__all__ = [
    "EMBEDDING_ERROR",
    "GENERATION_ERROR",
    "INVALID_REQUEST",
    "RETRIEVAL_ERROR",
    "RAGService",
    "rag_service",
]
