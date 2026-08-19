"""业务执行子 Agent：写操作 / 生成产物，内部按意图调用 Skill。"""
from __future__ import annotations

import asyncio
import time

from ecom_agent_matrix.config.constants import AGENT_EXEC
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.mcp.bus import mcp_bus
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.registry import register_agent
from ecom_agent_matrix.core.mcp.reply import build_reply
from ecom_agent_matrix.core.skill.skill_registry import skill_execution_context
from ecom_agent_matrix.modules.agent_cluster.handlers import (
    handle_ad,
    handle_crm,
    handle_report,
    handle_risk,
    handle_social,
)

logger = setup_logger("agent.exec")

_task_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _task_semaphore
    if _task_semaphore is None:
        _task_semaphore = asyncio.Semaphore(int(settings.EXEC_MAX_CONCURRENT))
    return _task_semaphore


def infer_exec_kind(payload: dict) -> str:
    """Exec Agent 内部意图：广告 / 报表 / 风控 / 社媒 / 客服答复。"""
    task_type = str(payload.get("task_type") or payload.get("exec_kind") or "").strip()
    if task_type in {"ad_optimize", "ad"}:
        return "ad"
    if task_type in {"ops_report", "report"}:
        return "report"
    if task_type in {"risk_control", "risk"}:
        return "risk"
    if task_type in {"social_marketing", "social"}:
        return "social"
    if task_type in {"customer_service", "crm"}:
        return "crm"

    text = " ".join(str(payload.get(k) or "") for k in ("query", "user_query", "text"))
    lower = text.lower()
    if any(k in text for k in ("风控", "触发风险")) or "risk" in lower:
        return "risk"
    if any(k in text for k in ("报表", "日报", "周报", "运营报告")) or "report" in lower:
        return "report"
    if any(k in text for k in ("广告", "出价", "投放优化", "ppc")) or any(
        k in lower for k in ("ad", "campaign", "bid")
    ):
        return "ad"
    if any(k in text for k in ("社媒", "文案", "tiktok", "instagram")) or "caption" in lower:
        return "social"
    if any(k in text for k in ("退款", "客服", "售后", "投诉")):
        return "crm"
    if payload.get("spend") is not None or payload.get("campaign_id"):
        return "ad"
    if payload.get("report_type"):
        return "report"
    if payload.get("run_risk_check") or payload.get("order_no"):
        return "risk"
    return "crm"


async def run_exec(payload: dict, *, task_id: str = "") -> tuple[bool, str, dict]:
    with skill_execution_context(AGENT_EXEC):
        kind = infer_exec_kind(payload)
        if kind == "ad":
            return await handle_ad(payload)
        if kind == "report":
            return await handle_report(payload)
        if kind == "risk":
            return await handle_risk(payload)
        if kind == "social":
            return await handle_social(payload)
        return await handle_crm(payload, task_id=task_id)


@register_agent(AGENT_EXEC)
async def exec_agent(msg_queue: asyncio.Queue):
    """业务执行：调出价、出报表、触发风控、生成文案/客服答复。"""
    logger.info(
        "exec_agent_started",
        extra={"event": "exec_agent_started", "agent": AGENT_EXEC},
    )
    sem = _get_semaphore()

    while True:
        msg: MCPMessage = await msg_queue.get()
        started = time.perf_counter()
        try:
            async with sem:
                payload = dict(msg.content or {})
                ok, err, data = await asyncio.wait_for(
                    run_exec(payload, task_id=msg.task_id),
                    timeout=float(settings.EXEC_SKILL_TIMEOUT),
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                data = {**(data or {}), "latency_ms": data.get("latency_ms") or round(elapsed_ms, 2)}
                reply = build_reply(
                    msg,
                    sender=AGENT_EXEC,
                    success=ok,
                    error_msg=err or "",
                    data=data,
                )
                await mcp_bus.send_msg(reply)
                logger.info(
                    "exec_task_done",
                    extra={
                        "event": "exec_task_done",
                        "task_id": msg.task_id,
                        "agent": AGENT_EXEC,
                        "query": data.get("exec_kind") or "",
                        "latency_ms": round(elapsed_ms, 2),
                    },
                )
        except asyncio.TimeoutError:
            reply = build_reply(
                msg,
                sender=AGENT_EXEC,
                success=False,
                error_msg=f"biz_exec 超时（>{settings.EXEC_SKILL_TIMEOUT}s）",
                data={},
            )
            await mcp_bus.send_msg(reply)
        except Exception as exc:
            logger.exception(
                "exec_task_failed",
                extra={
                    "event": "exec_task_failed",
                    "task_id": msg.task_id,
                    "agent": AGENT_EXEC,
                    "error": str(exc),
                },
            )
            reply = build_reply(
                msg,
                sender=AGENT_EXEC,
                success=False,
                error_msg=str(exc),
                data={},
            )
            await mcp_bus.send_msg(reply)
        finally:
            msg_queue.task_done()
