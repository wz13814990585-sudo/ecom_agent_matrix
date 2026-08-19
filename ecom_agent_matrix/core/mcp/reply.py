"""MCP 消息回传封装。"""
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.config.constants import MSG_PRIORITY_NORMAL


def build_reply(
    request: MCPMessage,
    sender: str,
    *,
    success: bool,
    data: dict | None = None,
    error_msg: str = "",
    msg_type: str = "agent_reply",
) -> MCPMessage:
    """
    封装 Agent 回传消息：保留 root task_id，并继承 request.correlation_id。
    """
    body = {
        "type": msg_type,
        "ref_task_id": request.task_id,
        "success": success,
        "request": request.content,
        "data": data or {},
    }
    if error_msg:
        body["error_msg"] = error_msg

    return MCPMessage(
        task_id=request.task_id,
        correlation_id=request.correlation_id,
        sender=sender,
        target=request.sender,
        priority=request.priority,
        content=body,
        security=request.security,
    )


def build_rag_reply(
    request: MCPMessage,
    *,
    query: str,
    lang: str,
    docs: list[dict],
    recall_count: int,
    latency_ms: float,
    cached: bool,
    success: bool = True,
    error_msg: str = "",
    answer: str = "",
    answer_source: str = "",
    llm_error: str = "",
    citations: list[dict] | None = None,
    grounded: bool = False,
    retrieval_version: str = "",
    error_code: str = "",
    invalid_citation_ids: list[str] | None = None,
    citation_status: str = "none",
    retrieval_mode: str = "none",
    degraded: bool = False,
    channel_errors: dict[str, str] | None = None,
    candidate_counts: dict[str, int] | None = None,
) -> MCPMessage:
    """RAG 检索 + 可选生成答复回传。"""
    from ecom_agent_matrix.config.constants import AGENT_RAG

    return build_reply(
        request,
        sender=AGENT_RAG,
        success=success,
        error_msg=error_msg,
        msg_type="rag_retrieve_result",
        data={
            "query": query,
            "lang": lang,
            "recall_count": recall_count,
            "latency_ms": round(latency_ms, 2),
            "cached": cached,
            "docs": docs,
            "answer": answer,
            "answer_source": answer_source,
            "llm_error": llm_error or None,
            "citations": citations or [],
            "grounded": grounded,
            "retrieval_version": retrieval_version,
            "error_code": error_code,
            "invalid_citation_ids": invalid_citation_ids or [],
            "citation_status": citation_status,
            "retrieval_mode": retrieval_mode,
            "degraded": degraded,
            "channel_errors": channel_errors or {},
            "candidate_counts": candidate_counts or {},
        },
    )
