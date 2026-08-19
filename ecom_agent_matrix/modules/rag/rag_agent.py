"""RAG MCP 子 Agent：检索 + 基于片段的 LLM 答复。"""
import asyncio
import time

from ecom_agent_matrix.config.constants import AGENT_RAG
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.mcp.bus import mcp_bus
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.registry import register_agent
from ecom_agent_matrix.core.mcp.reply import build_rag_reply
from ecom_agent_matrix.modules.rag.retriever import hybrid_retrieve
from ecom_agent_matrix.modules.skills.crm_reply import format_rag_docs
from ecom_agent_matrix.modules.utils.llm_explain import llm_explain

logger = setup_logger("rag.agent")

RAG_ANSWER_SYSTEM = (
    "你是跨境独立站知识库助手，只根据文档回答：店铺规则、运营手册、FAQ、商品说明。"
    "必须优先依据「检索片段」；不要查询或编造业务库中的订单、库存、广告消耗数字。"
    "若片段不足以回答，明确说明知识库暂无该信息。"
    "回答简洁，使用用户指定语种。"
)


def _fallback_answer(query: str, docs: list[dict], lang: str) -> str:
    if not docs:
        if lang.startswith("zh"):
            return f"未检索到与「{query}」相关的商品知识，请补充商品名或 SKU。"
        return f"No knowledge found for “{query}”. Please provide a product name or SKU."
    preview = format_rag_docs(docs, limit=2)
    if lang.startswith("zh"):
        return f"已检索到相关知识（LLM 未启用，仅展示片段）：\n{preview}"
    return f"Retrieved snippets (LLM disabled):\n{preview}"


@register_agent(AGENT_RAG)
async def rag_agent(msg_queue: asyncio.Queue):
    """RAG 智能体：消费 MCP 任务 → 检索 → LLM 生成答复 → 回传。"""
    logger.info(
        "rag_agent_started",
        extra={"event": "rag_agent_started", "agent": AGENT_RAG},
    )

    while True:
        msg: MCPMessage = await msg_queue.get()
        started = time.perf_counter()
        user_query = ""
        lang = "en"

        try:
            task_params = msg.content
            user_query = str(task_params.get("query", "")).strip()
            lang = str(task_params.get("lang", "en"))
            price_max = task_params.get("price_max")
            top_k = int(task_params.get("top_k", 8))

            if not user_query:
                reply = build_rag_reply(
                    msg,
                    query=user_query,
                    lang=lang,
                    docs=[],
                    recall_count=0,
                    latency_ms=0,
                    cached=False,
                    success=False,
                    error_msg="query 为空",
                )
                await mcp_bus.send_msg(reply)
                logger.warning(
                    "rag_empty_query",
                    extra={
                        "event": "rag_empty_query",
                        "task_id": msg.task_id,
                        "query": user_query,
                        "agent": AGENT_RAG,
                    },
                )
                continue

            logger.info(
                "rag_task_received",
                extra={
                    "event": "rag_task_received",
                    "task_id": msg.task_id,
                    "query": user_query,
                    "lang": lang,
                    "agent": AGENT_RAG,
                },
            )

            docs, cached, retrieve_ms = await hybrid_retrieve(
                user_query,
                lang,
                price_max,
                top_k,
                task_id=msg.task_id,
            )

            context = format_rag_docs(docs, limit=min(5, len(docs) or 0))
            fallback = _fallback_answer(user_query, docs, lang)
            answer, answer_source, llm_error = await llm_explain(
                system_prompt=RAG_ANSWER_SYSTEM,
                user_prompt=(
                    f"Language: {lang}\n"
                    f"User question: {user_query}\n\n"
                    f"Retrieved snippets:\n{context or '(empty)'}\n\n"
                    "Answer the question based on the snippets."
                ),
                fallback=fallback,
                max_tokens=int(getattr(settings, "AGENT_LLM_EXPLAIN_MAX_TOKENS", 450) or 450),
                temperature=0.2,
            )

            total_ms = (time.perf_counter() - started) * 1000
            reply = build_rag_reply(
                msg,
                query=user_query,
                lang=lang,
                docs=docs,
                recall_count=len(docs),
                latency_ms=total_ms,
                cached=cached,
                answer=answer,
                answer_source=answer_source,
                llm_error=llm_error,
            )
            await mcp_bus.send_msg(reply)

            logger.info(
                "rag_task_done",
                extra={
                    "event": "rag_task_done",
                    "task_id": msg.task_id,
                    "query": user_query,
                    "lang": lang,
                    "recall_count": len(docs),
                    "latency_ms": round(total_ms, 2),
                    "cached": cached,
                    "answer_source": answer_source,
                    "retrieve_ms": round(retrieve_ms, 2),
                    "agent": AGENT_RAG,
                },
            )

        except Exception as exc:
            total_ms = (time.perf_counter() - started) * 1000
            reply = build_rag_reply(
                msg,
                query=user_query,
                lang=lang,
                docs=[],
                recall_count=0,
                latency_ms=total_ms,
                cached=False,
                success=False,
                error_msg=str(exc),
            )
            await mcp_bus.send_msg(reply)
            logger.exception(
                "rag_task_failed",
                extra={
                    "event": "rag_task_failed",
                    "task_id": msg.task_id,
                    "query": user_query,
                    "latency_ms": round(total_ms, 2),
                    "agent": AGENT_RAG,
                    "error": str(exc),
                },
            )
        finally:
            msg_queue.task_done()
