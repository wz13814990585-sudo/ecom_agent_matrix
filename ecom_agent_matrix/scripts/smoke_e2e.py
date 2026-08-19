#!/usr/bin/env python3
"""端到端联调：MCP 进程内 或 HTTP 网关。

用法:
  python -m ecom_agent_matrix.scripts.smoke_e2e --mode social
  python -m ecom_agent_matrix.scripts.smoke_e2e --mode fast-path --transport http --api-key YOUR_KEY
  python -m ecom_agent_matrix.scripts.smoke_e2e --mode risk --transport http --api-key YOUR_KEY
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
    if mode == "fast-path":
        return (
            AGENT_MASTER,
            {
                "query": "查询 SKU-BAG-001 商品信息",
                "task_type": "goods_search",
                "sku": "SKU-BAG-001",
            },
            MSG_PRIORITY_NORMAL,
        )
    if mode == "rag":
        return (
            AGENT_MASTER,
            {
                "query": "防水户外背包有什么特点？",
                "user_query": "防水户外背包有什么特点？",
                "lang": "zh",
                "use_rag": True,
                "task_type": "knowledge_qa",
            },
            MSG_PRIORITY_CUSTOMER,
        )
    if mode == "composite":
        return (
            AGENT_MASTER,
            {
                "query": "根据 ORD-20260301-001 的订单状态和退款规则帮我回复客户",
            },
            MSG_PRIORITY_CUSTOMER,
        )
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
    if mode == "risk":
        return {
            "transport": "mcp",
            "success": True,
            "skipped": True,
            "error_msg": "risk approval demo requires authenticated HTTP transport",
            "data": {},
            "summary": "Use --transport http for the approval flow.",
        }
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

    if mode in {"customer", "rag"}:
        path = "/api/v1/customer/chat"
        body = {
            "query": (
                "防水户外背包有什么特点？"
                if mode == "rag" else "你好，我想咨询退款流程"
            ),
            "lang": "zh",
            "use_rag": mode == "rag",
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
    elif mode == "fast-path":
        path = "/api/v1/tasks"
        body = {
            "query": "查询 SKU-BAG-001 商品信息",
            "task_type": "goods_search",
            "payload": {"sku": "SKU-BAG-001"},
            "timeout": timeout,
        }
    elif mode == "composite":
        path = "/api/v1/tasks"
        body = {
            "query": "根据 ORD-20260301-001 的订单状态和退款规则帮我回复客户",
            "timeout": timeout,
        }
    elif mode == "risk":
        path = "/api/v1/tasks"
        body = {
            "query": "检查高风险订单 ORD-DEMO-RISK",
            "task_type": "risk_control",
            "payload": {
                "order_no": "ORD-DEMO-RISK",
                "total_amount": 501,
                "buy_count": 1,
            },
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

    async def post_json(session, request_path, *, request_headers, request_body=None):
        async with session.post(
            f"{base}{request_path}", headers=request_headers, json=request_body
        ) as resp:
            text = await resp.text()
            try:
                data = json.loads(text) if text else {}
            except json.JSONDecodeError:
                data = {"raw": text}
            return resp.status, data

    def find_approval_id(value):
        if isinstance(value, dict):
            if value.get("approval_id"):
                return str(value["approval_id"])
            for child in value.values():
                found = find_approval_id(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = find_approval_id(child)
                if found:
                    return found
        return ""

    timeout_cfg = aiohttp.ClientTimeout(total=timeout + 10)
    try:
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            http_status, data = await post_json(
                session, path, request_headers=headers, request_body=body
            )
            if mode == "risk":
                approval_id = find_approval_id(data)
                if http_status != 200 or not approval_id:
                    return {
                        "transport": "http", "path": path, "http_status": http_status,
                        "success": False, "error_msg": "approval was not created",
                        "summary": data.get("summary") or "", "data": data,
                    }
                approve_status, approve_data = await post_json(
                    session,
                    f"/api/v1/approvals/{approval_id}/approve",
                    request_headers=headers,
                )
                approved_headers = {**headers, "X-Approval-Id": approval_id}
                final_status, final_data = await post_json(
                    session, path, request_headers=approved_headers, request_body=body
                )
                return {
                    "transport": "http",
                    "path": path,
                    "http_status": final_status,
                    "success": (
                        approve_status == 200
                        and final_status == 200
                        and bool(final_data.get("success"))
                    ),
                    "error_msg": final_data.get("error_msg") or final_data.get("detail") or "",
                    "summary": final_data.get("summary") or "",
                    "data": {
                        "approval_id": approval_id,
                        "initial": data,
                        "approval": approve_data,
                        "resubmission": final_data,
                    },
                }
            return {
                "transport": "http",
                "http_status": http_status,
                "path": path,
                "success": http_status == 200 and bool(data.get("success")),
                "error_msg": data.get("error_msg") or data.get("detail") or "",
                "summary": data.get("summary") or "",
                "data": data,
            }
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return {
            "transport": "http", "path": path, "http_status": 0,
            "success": False, "error_msg": f"dependency unavailable: {type(exc).__name__}",
            "summary": "", "data": {},
        }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ecom Agent Matrix smoke e2e")
    parser.add_argument(
        "--mode",
        choices=[
            "fast-path", "rag", "composite", "risk",
            "social", "competitor", "customer",
        ],
        default="fast-path",
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
