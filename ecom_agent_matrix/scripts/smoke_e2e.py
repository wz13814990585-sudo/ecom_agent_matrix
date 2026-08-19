#!/usr/bin/env python3
"""端到端联调：MCP 进程内 或 HTTP 网关。

用法:
  python -m ecom_agent_matrix.scripts.smoke_e2e --mode social
  python -m ecom_agent_matrix.scripts.smoke_e2e --mode competitor
  python -m ecom_agent_matrix.scripts.smoke_e2e --mode customer
  python -m ecom_agent_matrix.scripts.smoke_e2e --mode social --transport http --base-url http://127.0.0.1:8000
  python -m ecom_agent_matrix.scripts.smoke_e2e --check-deps
"""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid

import ecom_agent_matrix.modules.agent_cluster  # noqa: F401
import ecom_agent_matrix.modules.skills  # noqa: F401
from ecom_agent_matrix.api.health import readiness_report
from ecom_agent_matrix.config.constants import (
    AGENT_MASTER,
    MSG_PRIORITY_CUSTOMER,
    MSG_PRIORITY_NORMAL,
    MSG_PRIORITY_RISK,
)
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.mcp.bus import mcp_bus
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.registry import agent_map, start_all_agents
from ecom_agent_matrix.core.mcp.result_waiter import GatewayResultWaiter

logger = setup_logger("smoke_e2e")


def _build_payload(mode: str) -> tuple[str, dict, int]:
    """返回 (target_or_path_hint, content, priority)。"""
    if mode == "competitor":
        return (
            AGENT_MASTER,
            {
                "query": "监控 Temu 上 SKU-BAG-001 的价格",
                "task_type": "competitor_watch",
                "sku": "SKU-BAG-001",
                "target_sku": "SKU-BAG-001",
                "competitor": "Temu",
            },
            MSG_PRIORITY_RISK,
        )
    if mode == "customer":
        return (
            AGENT_MASTER,
            {
                "query": "你好，我想咨询退款流程",
                "user_query": "你好，我想咨询退款流程",
                "lang": "zh",
                "use_rag": True,
                "task_type": "knowledge_qa",
                "task_type": "knowledge_qa",
            },
            MSG_PRIORITY_CUSTOMER,
        )
    return (
        AGENT_MASTER,
        {
            "query": "为「防水户外背包」生成tiktok文案",
            "task_type": "social_marketing",
            "product_name": "防水户外背包",
            "platform": "tiktok",
        },
        MSG_PRIORITY_NORMAL,
    )


async def _run_mcp(mode: str, timeout: float) -> dict:
    target, content, priority = _build_payload(mode)
    task_id = str(uuid.uuid4())
    GatewayResultWaiter.begin(task_id)
    msg = MCPMessage(
        task_id=task_id,
        sender=settings.API_SENDER,
        target=target,
        priority=priority,
        content=content,
    )
    await mcp_bus.send_msg(msg)
    reply = await GatewayResultWaiter.wait(task_id, timeout)
    if reply is None:
        return {"task_id": task_id, "success": False, "error_msg": "timeout", "data": {}, "summary": ""}
    data = reply.content.get("data") or {}
    summary = ""
    if isinstance(data, dict):
        summary = str(data.get("summary") or "")
    return {
        "task_id": task_id,
        "transport": "mcp",
        "target": target,
        "success": bool(reply.content.get("success")),
        "error_msg": reply.content.get("error_msg") or "",
        "data": data,
        "msg_type": reply.content.get("type"),
        "summary": summary,
    }


async def _run_http(mode: str, base_url: str, timeout: float, api_key: str) -> dict:
    import aiohttp

    base = base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    elif settings.API_KEY:
        headers["X-API-Key"] = settings.API_KEY

    if mode == "customer":
        path = "/api/v1/customer/chat"
        body = {
            "query": "你好，我想咨询退款流程",
            "lang": "zh",
            "use_rag": False,
            "timeout": timeout,
        }
    elif mode == "competitor":
        path = "/api/v1/warn/competitor"
        body = {
            "query": "监控 Temu 上 SKU-BAG-001 的价格",
            "sku": "SKU-BAG-001",
            "competitor": "Temu",
            "via_master": True,
            "timeout": timeout,
        }
    else:
        path = "/api/v1/tasks"
        body = {
            "query": "为「防水户外背包」生成tiktok文案",
            "task_type": "social_marketing",
            "payload": {"product_name": "防水户外背包", "platform": "tiktok"},
            "timeout": timeout,
        }

    timeout_cfg = aiohttp.ClientTimeout(total=timeout + 10)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.post(f"{base}{path}", headers=headers, json=body) as resp:
            text = await resp.text()
            try:
                data = json.loads(text) if text else {}
            except json.JSONDecodeError:
                data = {"raw": text}
            return {
                "transport": "http",
                "http_status": resp.status,
                "path": path,
                "success": resp.status == 200 and bool(data.get("success")),
                "error_msg": data.get("error_msg") or data.get("detail") or "",
                "summary": data.get("summary") or "",
                "data": data,
            }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ecom Agent Matrix smoke e2e")
    parser.add_argument(
        "--mode",
        choices=["social", "competitor", "customer"],
        default="social",
    )
    parser.add_argument("--transport", choices=["mcp", "http"], default="mcp")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="仅探测 Postgres/Redis 后退出",
    )
    args = parser.parse_args()

    if args.check_deps:
        report = await readiness_report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        try:
            from ecom_agent_matrix.db.base import AsyncPGClient
            from ecom_agent_matrix.db.redis_client import AsyncRedisClient

            await AsyncPGClient.close()
            await AsyncRedisClient.close()
        except Exception:
            pass
        raise SystemExit(0 if report.get("ready") else 2)

    if args.transport == "http":
        result = await _run_http(args.mode, args.base_url, args.timeout, args.api_key)
        if result.get("summary"):
            print("=== 可读摘要 ===")
            print(result["summary"])
            print("=== 完整结果 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        raise SystemExit(0 if result.get("success") else 1)

    if not agent_map:
        raise SystemExit("agent_map 为空，侧载注册失败")

    agent_task = asyncio.create_task(start_all_agents(), name="agents")
    await asyncio.sleep(0.1)
    try:
        result = await _run_mcp(args.mode, args.timeout)
        if result.get("summary"):
            print("=== 可读摘要 ===")
            print(result["summary"])
            print("=== 完整结果 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        if not result.get("success"):
            raise SystemExit(1)
    finally:
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        try:
            from ecom_agent_matrix.db.base import AsyncPGClient
            from ecom_agent_matrix.db.redis_client import AsyncRedisClient

            await AsyncPGClient.close()
            await AsyncRedisClient.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
