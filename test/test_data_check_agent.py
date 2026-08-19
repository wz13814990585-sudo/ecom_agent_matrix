"""数据校验 Agent 测试：解析辅助 + MCP 端到端（打印具体过程）。"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ecom_agent_matrix.modules.agent_cluster  # noqa: F401
import ecom_agent_matrix.modules.skills  # noqa: F401
from ecom_agent_matrix.config.constants import AGENT_EXEC, AGENT_QUERY, MSG_PRIORITY_NORMAL
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.mcp.bus import mcp_bus
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.registry import agent_map, start_all_agents
from ecom_agent_matrix.core.mcp.result_waiter import GatewayResultWaiter
from ecom_agent_matrix.core.skill.base_skill import SkillResult
from ecom_agent_matrix.modules.agent_cluster.handlers.data_check import (
    _extract_order_no,
    _extract_scope,
)
from ecom_agent_matrix.modules.utils.competitor_parse import extract_sku


def banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_extract_order_no():
    banner("1) _extract_order_no 订单号解析")
    cases = [
        ({"order_no": "ORD-20260816-001"}, "ORD-20260816-001"),
        ({"query": "查一下订单 ORD_ABC_99 状态"}, "ORD_ABC_99"),
        ({"query": "淘宝单号 1234567890123456"}, "1234567890123456"),
        ({"query": "没有单号"}, ""),
    ]
    for payload, expect in cases:
        got = _extract_order_no(payload)
        print(f"输入={payload}")
        print(f"  → order_no={got!r}")
        assert got == expect
    print("订单号解析 OK")


def test_extract_scope():
    banner("2) _extract_scope 校验范围")
    cases = [
        ({"scope": "goods"}, "goods"),
        ({"query": "检查订单数据完整性"}, "order"),
        ({"query": "商品主数据校验"}, "goods"),
        ({"query": "做一次全面数据检查"}, "full"),
        ({"query": "订单风控扫描"}, "order"),
    ]
    for payload, expect in cases:
        got = _extract_scope(payload)
        print(f"输入={payload}")
        print(f"  → scope={got!r}")
        assert got == expect
    print("scope 解析 OK")


def test_extract_sku_helper():
    banner("3) extract_sku（Agent 共用）")
    payload = {"query": "校验 SKU-BAG-001 商品字段"}
    sku = extract_sku(payload)
    print(f"输入={payload}")
    print(f"  → sku={sku!r}")
    assert sku == "SKU-BAG-001"
    print("SKU 解析 OK")


async def _dispatch(target: str, content: dict, timeout: float = 25.0) -> dict:
    task_id = str(uuid.uuid4())
    GatewayResultWaiter.begin(task_id)
    msg = MCPMessage(
        task_id=task_id,
        sender=settings.API_SENDER,
        target=target,
        priority=MSG_PRIORITY_NORMAL,
        content=content,
    )
    await mcp_bus.send_msg(msg)
    reply = await GatewayResultWaiter.wait(task_id, timeout)
    if reply is None:
        return {"success": False, "error_msg": "timeout", "data": {}}
    body = reply.content or {}
    return {
        "success": bool(body.get("success")),
        "error_msg": body.get("error_msg") or "",
        "data": body.get("data") or {},
        "reply_from": reply.sender,
    }


async def test_data_check_agent_e2e_mocked():
    banner("4) MCP 端到端（Mock Skill，打印编排结果）")

    if AGENT_QUERY not in agent_map:
        raise SystemExit(f"Agent 未注册: {AGENT_QUERY}")

    async def fake_exec_skill(name: str, params: dict):
        print(f"\n  [exec_skill] name={name}")
        print(f"  [exec_skill] params={json.dumps(params, ensure_ascii=False, default=str)}")

        if name == "data_integrity_check":
            scope = params.get("scope") or "full"
            issues = []
            if scope in ("goods", "full") and params.get("sku") == "SKU-BAD-001":
                issues.append(
                    {
                        "entity": "goods",
                        "sku": "SKU-BAD-001",
                        "problems": ["price_invalid", "title_missing"],
                        "price": 0,
                        "stock_num": 3,
                    }
                )
            if scope in ("order", "full") and params.get("order_no"):
                # 默认无问题
                pass
            passed = len(issues) == 0
            return SkillResult(
                success=True,
                data={
                    "scope": scope,
                    "passed": passed,
                    "issue_count": len(issues),
                    "issues": issues,
                    "checked": {"sku": params.get("sku"), "order_no": params.get("order_no")},
                },
            )

        if name == "order_risk_check":
            amount = float(params.get("total_amount") or 0)
            count = int(params.get("buy_count") or 0)
            tags = []
            if amount > 500:
                tags.append("大额订单")
            if count > 20:
                tags.append("批量囤货")
            return SkillResult(
                success=True,
                data={
                    "is_risk": bool(tags),
                    "risk_detail": "、".join(tags) if tags else "无异常风险",
                },
            )

        if name == "safe_sql_query":
            sql = str(params.get("sql") or "")
            if "DELETE" in sql.upper():
                return SkillResult(success=False, error_msg="仅允许执行SELECT只读查询")
            return SkillResult(
                success=True,
                data={"rows": [{"sku": "SKU-BAG-001", "stock_num": 12}], "row_count": 1},
            )

        return SkillResult(success=False, error_msg=f"unexpected skill: {name}")

    agent_task = asyncio.create_task(start_all_agents(), name="agents")
    await asyncio.sleep(0.15)

    with patch(
        "ecom_agent_matrix.modules.agent_cluster.handlers.data_check.exec_skill",
        new=AsyncMock(side_effect=fake_exec_skill),
    ):
        try:
            # 4.1 商品校验发现问题
            print("\n--- 4.1 商品校验：SKU-BAD-001 有问题 ---")
            r1 = await _dispatch(
                AGENT_QUERY,
                {
                    "query": "检查商品主数据 SKU-BAD-001",
                    "sku": "SKU-BAD-001",
                    "scope": "goods",
                    "task_type": "data_check",
                },
            )
            print(json.dumps(r1, ensure_ascii=False, indent=2, default=str))
            assert r1["success"] is True
            assert r1["reply_from"] == AGENT_QUERY
            assert r1["data"].get("data_ok") is False
            assert "发现" in (r1["error_msg"] or "")
            print("\n[integrity.issues]")
            print(json.dumps(r1["data"].get("integrity", {}).get("issues"), ensure_ascii=False, indent=2))

            # 4.2 订单校验通过
            print("\n--- 4.2 订单校验：ORD-OK-001 通过 ---")
            r2 = await _dispatch(
                AGENT_QUERY,
                {
                    "query": "校验订单 ORD-OK-001",
                    "order_no": "ORD-OK-001",
                    "scope": "order",
                    "task_type": "data_check",
                },
            )
            print(json.dumps(r2, ensure_ascii=False, indent=2, default=str))
            assert r2["success"] is True
            assert r2["data"].get("data_ok") is True
            assert r2["data"].get("scope") == "order"

            # 4.3 风控：缺金额时 skipped
            print("\n--- 4.3 触发风控但缺 total_amount/buy_count → skipped ---")
            r3 = await _dispatch(
                AGENT_EXEC,
                {
                    "query": "订单风控 ORD-RISK-1",
                    "order_no": "ORD-RISK-1",
                    "run_risk_check": True,
                    "task_type": "risk_control",
                },
            )
            print(json.dumps(r3, ensure_ascii=False, indent=2, default=str))
            assert r3["data"].get("risk", {}).get("skipped") is True
            print("[risk]", r3["data"].get("risk"))

            # 4.4 风控命中大额+囤货
            print("\n--- 4.4 风控命中：大额 + 批量 ---")
            r4 = await _dispatch(
                AGENT_EXEC,
                {
                    "query": "订单风控检查",
                    "order_no": "ORD-RISK-2",
                    "run_risk_check": True,
                    "total_amount": 888,
                    "buy_count": 30,
                    "task_type": "risk_control",
                },
            )
            print(json.dumps(r4, ensure_ascii=False, indent=2, default=str))
            risk = (r4["data"].get("risk") or {}).get("data") or {}
            print("[risk.data]", risk)
            assert risk.get("is_risk") is True
            assert "大额" in (risk.get("risk_detail") or "")

            # 4.5 自定义只读 SQL
            print("\n--- 4.5 safe_sql_query 只读查询 ---")
            r5 = await _dispatch(
                AGENT_QUERY,
                {
                    "query": "全面数据检查",
                    "scope": "full",
                    "sql": "SELECT sku, stock_num FROM ecom_goods LIMIT 1",
                    "task_type": "data_check",
                },
            )
            print(json.dumps(r5, ensure_ascii=False, indent=2, default=str))
            sql = r5["data"].get("sql") or {}
            assert sql.get("skipped") is False
            assert sql.get("success") is True
            print("[sql.data]", sql.get("data"))

            # 4.6 危险 SQL 被拒
            print("\n--- 4.6 危险 SQL 应失败（仍返回 integrity 成功）---")
            r6 = await _dispatch(
                AGENT_QUERY,
                {
                    "scope": "goods",
                    "sku": "SKU-BAG-001",
                    "sql": "DELETE FROM ecom_goods WHERE id=1",
                    "task_type": "data_check",
                },
            )
            print(json.dumps(r6, ensure_ascii=False, indent=2, default=str))
            assert r6["success"] is True  # integrity 成功
            assert r6["data"].get("data_ok") is True
            assert "sql:" in (r6["error_msg"] or "")
            print("[sql.error]", (r6["data"].get("sql") or {}).get("error_msg"))

        finally:
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
            # 不在此处关闭 PG：后面还有 live 段可能复用连接池

    print("\nMock 端到端 OK")


async def test_data_check_live_pg():
    """有 Postgres 时打一次真实 integrity check；失败则跳过。"""
    banner("5) Live Postgres：真实 data_integrity_check（可选）")
    try:
        from ecom_agent_matrix.api.health import readiness_report

        report = await readiness_report()
        print("readiness:", json.dumps(report, ensure_ascii=False))
        if not report.get("ready"):
            print("依赖未就绪，跳过 live")
            return
    except Exception as exc:
        print("readiness 探测失败，跳过 live:", exc)
        return

    agent_task = asyncio.create_task(start_all_agents(), name="agents-live")
    await asyncio.sleep(0.15)
    try:
        r = await _dispatch(
            AGENT_QUERY,
            {
                "query": "全面数据检查",
                "scope": "full",
                "limit": 10,
                "task_type": "data_check",
            },
            timeout=30.0,
        )
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
        integrity = r.get("data", {}).get("integrity") or {}
        print("\n[live] data_ok =", r.get("data", {}).get("data_ok"))
        print("[live] issue_count =", integrity.get("issue_count"))
        issues = integrity.get("issues") or []
        if issues:
            print("[live] 前 3 条 issues:")
            print(json.dumps(issues[:3], ensure_ascii=False, indent=2, default=str))
        else:
            print("[live] 未发现问题（或库为空）")
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


async def main():
    test_extract_order_no()
    test_extract_scope()
    test_extract_sku_helper()
    await test_data_check_agent_e2e_mocked()
    await test_data_check_live_pg()
    print("\n" + "=" * 60)
    print("data_check_agent 测试流程结束")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
