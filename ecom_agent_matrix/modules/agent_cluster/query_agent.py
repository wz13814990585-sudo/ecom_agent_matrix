"""数据查询子 Agent：只读查询，内部按意图调用 DB Skill。"""
from __future__ import annotations

import asyncio
import time

from ecom_agent_matrix.config.constants import AGENT_QUERY
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.mcp.bus import mcp_bus
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.registry import register_agent
from ecom_agent_matrix.core.mcp.reply import build_reply
from ecom_agent_matrix.modules.agent_cluster.handlers import (
    handle_data_check,
    handle_goods,
    handle_price_warn,
    handle_stock,
)
from ecom_agent_matrix.modules.skills.goods_catalog import is_catalog_query
from ecom_agent_matrix.modules.utils.competitor_parse import extract_sku

logger = setup_logger("agent.query")

_task_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _task_semaphore
    if _task_semaphore is None:
        _task_semaphore = asyncio.Semaphore(int(settings.QUERY_MAX_CONCURRENT))
    return _task_semaphore


def infer_query_kind(payload: dict) -> str:
    """Query Agent 内部意图：商品 / 库存 / 竞品 / 数据校验。不对外暴露为独立 Agent。"""
    task_type = str(payload.get("task_type") or payload.get("query_kind") or "").strip()
    if task_type in {"goods_catalog", "goods_search"}:
        return "goods"
    if task_type in {"stock_analysis", "stock"}:
        return "stock"
    if task_type in {"competitor_watch", "competitor"}:
        return "competitor"
    if task_type in {"data_check", "order_query", "ad_query"}:
        return "data_check"

    text = " ".join(
        str(payload.get(k) or "")
        for k in ("query", "user_query", "text", "product_name")
    )
    lower = text.lower()
    if is_catalog_query(text) or payload.get("mode") == "catalog":
        return "goods"
    if any(k in text for k in ("竞品", "比价", "竞价对比", "价格对比", "跟价")) or "competitor" in lower:
        return "competitor"
    if any(k in text for k in ("库存", "备货", "补货", "缺货")) or any(
        k in lower for k in ("stock", "inventory", "replenish")
    ):
        return "stock"
    if any(
        k in text
        for k in ("数据校验", "数据检查", "完整性", "脏数据", "主数据", "查库", "跑sql", "订单", "校验")
    ) or any(k in lower for k in ("integrity", "sql", "order")):
        return "data_check"
    if any(k in text for k in ("广告数据", "投放数据", "广告消耗")):
        return "data_check"
    return "goods"


async def _ensure_sku(payload: dict) -> tuple[dict, dict | None]:
    """缺 SKU 时先走商品检索；成功则把 best_sku 写回 payload。"""
    if extract_sku(payload):
        return payload, None
    ok, err, goods_data = await handle_goods(payload)
    if not ok or not (goods_data or {}).get("best_sku"):
        return payload, {
            "success": ok,
            "error_msg": err or "未找到匹配商品，无法继续查询",
            "data": {**(goods_data or {}), "query_kind": goods_data.get("query_kind") if goods_data else "goods"},
        }
    sku = goods_data["best_sku"]
    merged = {
        **payload,
        "sku": sku,
        "best_sku": sku,
        "target_sku": sku,
        "candidates": goods_data.get("candidates") or [],
        "_goods": goods_data,
    }
    return merged, None


async def run_query(payload: dict) -> tuple[bool, str, dict]:
    """执行一次只读查询（可供单测直接调用）。"""
    kind = infer_query_kind(payload)
    if kind == "goods":
        return await handle_goods(payload)

    if kind == "stock":
        merged, early = await _ensure_sku(payload)
        if early:
            return early["success"], early["error_msg"], early["data"]
        return await handle_stock(merged)

    if kind == "competitor":
        merged, early = await _ensure_sku(payload)
        if early:
            return early["success"], early["error_msg"], early["data"]
        if not merged.get("competitor") and not merged.get("multi_compare"):
            merged = {**merged, "multi_compare": True}
        return await handle_price_warn(merged)

    return await handle_data_check(payload)


@register_agent(AGENT_QUERY)
async def query_agent(msg_queue: asyncio.Queue):
    """数据查询：广告/订单/库存/竞品/商品目录，只调只读 Skill。"""
    logger.info(
        "query_agent_started",
        extra={"event": "query_agent_started", "agent": AGENT_QUERY},
    )
    sem = _get_semaphore()

    while True:
        msg: MCPMessage = await msg_queue.get()
        started = time.perf_counter()
        try:
            async with sem:
                payload = dict(msg.content or {})
                ok, err, data = await asyncio.wait_for(
                    run_query(payload),
                    timeout=float(settings.QUERY_SKILL_TIMEOUT),
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                data = {**(data or {}), "latency_ms": data.get("latency_ms") or round(elapsed_ms, 2)}
                reply = build_reply(
                    msg,
                    sender=AGENT_QUERY,
                    success=ok,
                    error_msg=err or "",
                    data=data,
                )
                await mcp_bus.send_msg(reply)
                logger.info(
                    "query_task_done",
                    extra={
                        "event": "query_task_done",
                        "task_id": msg.task_id,
                        "agent": AGENT_QUERY,
                        "query": data.get("query_kind") or "",
                        "latency_ms": round(elapsed_ms, 2),
                    },
                )
        except asyncio.TimeoutError:
            reply = build_reply(
                msg,
                sender=AGENT_QUERY,
                success=False,
                error_msg=f"data_query 超时（>{settings.QUERY_SKILL_TIMEOUT}s）",
                data={},
            )
            await mcp_bus.send_msg(reply)
        except Exception as exc:
            logger.exception(
                "query_task_failed",
                extra={
                    "event": "query_task_failed",
                    "task_id": msg.task_id,
                    "agent": AGENT_QUERY,
                    "error": str(exc),
                },
            )
            reply = build_reply(
                msg,
                sender=AGENT_QUERY,
                success=False,
                error_msg=str(exc),
                data={},
            )
            await mcp_bus.send_msg(reply)
        finally:
            msg_queue.task_done()
