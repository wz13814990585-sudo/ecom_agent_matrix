"""RAG MCP Agent：thin RAGService input/output adapter。"""
from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError

from ecom_agent_matrix.config.constants import AGENT_RAG
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.mcp.bus import mcp_bus
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.registry import register_agent
from ecom_agent_matrix.core.mcp.reply import build_rag_reply
from ecom_agent_matrix.modules.rag.schemas import RAGRequest
from ecom_agent_matrix.modules.rag.service import rag_service
from ecom_agent_matrix.core.security import require_trusted_ingress
from ecom_agent_matrix.core.security import tenant_scope_from_security
from ecom_agent_matrix.platform.observability.context import TraceContext, set_trace_context
from ecom_agent_matrix.platform.observability.metrics import metrics

logger = setup_logger("rag.agent")


def _legacy_document(document) -> dict:
    value = document.model_dump()
    value["meta"] = value.pop("metadata", {})
    return value


@register_agent(AGENT_RAG)
async def rag_agent(msg_queue: asyncio.Queue):
    """Convert MCP payload to RAGRequest and return a legacy-compatible MCP reply."""
    logger.info("rag_agent_started", extra={"event": "rag_agent_started", "agent": AGENT_RAG})
    while True:
        msg: MCPMessage = await msg_queue.get()
        started = time.perf_counter()
        set_trace_context(TraceContext.from_identity(
            task_id=msg.task_id, correlation_id=msg.correlation_id, agent_id=AGENT_RAG,
            workflow="rag_answer", tenant_id=getattr(msg.security, "tenant_id", ""),
            user_id=getattr(msg.security, "user_id", ""),
        ))
        query = ""
        lang = "en"
        try:
            require_trusted_ingress(msg.security, app_env=settings.APP_ENV)
            payload = dict(msg.content or {})
            query = str(payload.get("query") or "").strip()
            lang = str(payload.get("lang") or "en")
            request = RAGRequest(
                query=query,
                lang=lang,
                price_max=payload.get("price_max"),
                top_k=payload.get("top_k", 8),
                task_id=msg.task_id,
            )
            result = await rag_service.answer(
                request, scope=tenant_scope_from_security(msg.security)
            )
            reply = build_rag_reply(
                msg,
                query=request.query,
                lang=request.lang,
                docs=[_legacy_document(document) for document in result.documents],
                recall_count=len(result.documents),
                latency_ms=result.total_latency_ms,
                cached=result.cached,
                success=result.success,
                error_msg=result.error_msg,
                error_code=result.error_code,
                answer=result.answer,
                answer_source=result.answer_source,
                llm_error=result.error_code if result.error_code == "GENERATION_ERROR" else "",
                citations=[citation.model_dump() for citation in result.citations],
                grounded=result.grounded,
                retrieval_version=settings.RAG_RETRIEVAL_VERSION,
                invalid_citation_ids=result.invalid_citation_ids,
                citation_status=result.citation_status,
                retrieval_mode=result.retrieval_mode,
                degraded=result.degraded,
                channel_errors=result.channel_errors,
                candidate_counts=result.candidate_counts,
            )
            await mcp_bus.send_msg(reply)
            metrics.observe_agent(AGENT_RAG, result.success, time.perf_counter() - started)
            logger.info(
                "rag_task_done",
                extra={
                    "event": "rag_task_done",
                    "task_id": msg.task_id,
                    "recall_count": len(result.documents),
                    "latency_ms": result.total_latency_ms,
                    "cached": result.cached,
                    "answer_source": result.answer_source,
                    "agent": AGENT_RAG,
                },
            )
        except (ValidationError, TypeError, ValueError) as exc:
            metrics.observe_agent(AGENT_RAG, False, time.perf_counter() - started)
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            await mcp_bus.send_msg(
                build_rag_reply(
                    msg,
                    query=query,
                    lang=lang,
                    docs=[],
                    recall_count=0,
                    latency_ms=elapsed,
                    cached=False,
                    success=False,
                    error_msg="Invalid RAG request",
                    error_code="INVALID_REQUEST",
                    retrieval_version=settings.RAG_RETRIEVAL_VERSION,
                )
            )
            logger.warning(
                "rag_request_invalid",
                extra={
                    "event": "rag_request_invalid",
                    "task_id": msg.task_id,
                    "error_type": type(exc).__name__,
                    "latency_ms": elapsed,
                },
            )
        except Exception as exc:
            metrics.observe_agent(AGENT_RAG, False, time.perf_counter() - started)
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            await mcp_bus.send_msg(
                build_rag_reply(
                    msg,
                    query=query,
                    lang=lang,
                    docs=[],
                    recall_count=0,
                    latency_ms=elapsed,
                    cached=False,
                    success=False,
                    error_msg="RAG retrieval failed",
                    error_code="RETRIEVAL_ERROR",
                    retrieval_version=settings.RAG_RETRIEVAL_VERSION,
                )
            )
            logger.error(
                "rag_task_failed",
                extra={
                    "event": "rag_task_failed",
                    "task_id": msg.task_id,
                    "error_type": type(exc).__name__,
                    "latency_ms": elapsed,
                    "agent": AGENT_RAG,
                },
            )
        finally:
            msg_queue.task_done()
