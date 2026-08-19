"""HTTP → MCP 下发与等待。"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import HTTPException, status

from ecom_agent_matrix.config.constants import AGENT_MASTER
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm.output_polish import polish_final_output
from ecom_agent_matrix.core.mcp.bus import mcp_bus
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.registry import agent_map
from ecom_agent_matrix.core.mcp.result_waiter import GatewayResultWaiter


async def dispatch_and_wait(
    *,
    target: str,
    content: dict[str, Any],
    priority: int,
    timeout: float | None = None,
) -> dict[str, Any]:
    """向目标 Agent 发任务并等待最终回传，并生成可读 summary。"""
    if target not in agent_map:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent 未注册: {target}",
        )

    task_id = str(uuid.uuid4())
    wait_timeout = float(timeout if timeout is not None else settings.API_REQUEST_TIMEOUT)
    GatewayResultWaiter.begin(task_id)
    msg = MCPMessage(
        task_id=task_id,
        sender=settings.API_SENDER,
        target=target,
        priority=priority,
        content=content,
    )
    try:
        await mcp_bus.send_msg(msg)
        reply = await GatewayResultWaiter.wait(task_id, wait_timeout)
    except Exception:
        GatewayResultWaiter.cancel(task_id)
        raise

    if reply is None:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"等待 {target} 回传超时（>{wait_timeout}s）",
            headers={"X-Task-Id": task_id},
        )

    body = reply.content or {}
    data = body.get("data") or {}
    if not isinstance(data, dict):
        data = {"raw": data}
    success = bool(body.get("success"))
    error_msg = body.get("error_msg") or ""
    user_query = str(
        content.get("query")
        or content.get("user_query")
        or content.get("product_name")
        or ""
    )

    # Master 若已在 data.summary 写好，直接复用；否则统一整理
    if isinstance(data.get("summary"), str) and data["summary"].strip():
        summary = data["summary"].strip()
    else:
        summary = await polish_final_output(
            success=success,
            data=data,
            error_msg=error_msg,
            user_query=user_query,
            reply_from=reply.sender or target,
        )

    return {
        "task_id": task_id,
        "target": target,
        "reply_from": reply.sender,
        "success": success,
        "data": data,
        "error_msg": error_msg,
        "msg_type": body.get("type") or "",
        "summary": summary,
    }


async def dispatch_to_master(
    content: dict[str, Any],
    *,
    priority: int,
    timeout: float | None = None,
) -> dict[str, Any]:
    return await dispatch_and_wait(
        target=AGENT_MASTER,
        content=content,
        priority=priority,
        timeout=timeout,
    )
