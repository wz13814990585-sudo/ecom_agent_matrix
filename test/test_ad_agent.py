"""广告优化 Agent 测试：解析辅助 + MCP 端到端（打印具体过程）。"""
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
from ecom_agent_matrix.config.constants import AGENT_EXEC, MSG_PRIORITY_AD
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.mcp.bus import mcp_bus
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.registry import agent_map, start_all_agents
from ecom_agent_matrix.core.mcp.result_waiter import GatewayResultWaiter
from ecom_agent_matrix.modules.agent_cluster.handlers.ad import (
    _build_skill_params,
    _extract_platform,
    _parse_metrics_from_query,
)


def banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_extract_platform():
    banner("1) _extract_platform 平台解析")
    cases = [
        ({"platform": "fb"}, "meta"),
        ({"platform": "Google Ads"}, "google"),
        ({"query": "优化 tiktok ads 投放"}, "tiktok"),
        ({"query": "没有平台信息"}, "meta"),  # 默认
        ({"platform": "unknown_xyz"}, None),
    ]
    for payload, expect in cases:
        platform, err = _extract_platform(payload)
        print(f"输入={payload}")
        print(f"  → platform={platform!r} err={err!r}")
        if expect is None:
            assert platform is None and err
        else:
            assert platform == expect and not err
    print("平台解析 OK")


def test_parse_metrics_from_query():
    banner("2) _parse_metrics_from_query 从自然语言抽指标")
    text = "消耗 120.5，成交 360，转化 12，超级大的点击 800"
    out = _parse_metrics_from_query(text)
    print(f"输入: {text}")
    print(f"输出: {json.dumps(out, ensure_ascii=False)}")
    assert out["spend"] == 120.5
    assert out["revenue"] == 360.0
    assert out["clicks"] == 800.0
    assert out["conversions"] == 12.0
    print("指标抽取 OK")


def test_build_skill_params():
    banner("3) _build_skill_params 组装 skill 入参")
    payload = {
        "sku": "SKU-BAG-001",
        "campaign_id": "cmp-001",
        "query": "消耗 80，营收 200，点击 500",
        "_platform": "meta",
        "daily_budget": 50,
        "bid": 0.8,
        "target_roas": 2.5,
    }
    params = _build_skill_params(payload)
    print(json.dumps(params, ensure_ascii=False, indent=2))
    assert params["sku"] == "SKU-BAG-001"
    assert params["spend"] == 80.0
    assert params["revenue"] == 200.0
    assert params["clicks"] == 500.0
    assert params["platform"] == "meta"
    assert params["daily_budget"] == 50.0
    print("参数组装 OK")


async def _dispatch_ad(content: dict, timeout: float = 25.0) -> dict:
    """向 ad_optimizer 发任务并等待回传。"""
    task_id = str(uuid.uuid4())
    GatewayResultWaiter.begin(task_id)
    msg = MCPMessage(
        task_id=task_id,
        sender=settings.API_SENDER,
        target=AGENT_EXEC,
        priority=MSG_PRIORITY_AD,
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


async def test_ad_agent_e2e():
    banner("4) MCP 端到端：缺数据 / 非法平台 / 正常优化 / 附带利润")

    if AGENT_EXEC not in agent_map:
        raise SystemExit(f"Agent 未注册: {AGENT_EXEC}")

    # 避免依赖 Postgres 向量记忆（须在 Agent 启动前 patch）
    mem_patch = patch(
        "ecom_agent_matrix.modules.agent_cluster.handlers.ad.AgentLongVectorMemory",
    )
    mock_mem_cls = mem_patch.start()
    mock_mem = mock_mem_cls.return_value
    mock_mem.recall = AsyncMock(return_value=[])
    mock_mem.safe_save_memory = AsyncMock(return_value=True)

    # 强制走规则兜底，结果可复现（不依赖外网）
    old_key = settings.DEEPSEEK_API_KEY
    settings.DEEPSEEK_API_KEY = ""

    agent_task = asyncio.create_task(start_all_agents(), name="agents")
    await asyncio.sleep(0.15)

    try:
        # 4.1 缺少投放数据
        print("\n--- 4.1 缺少投放数据 ---")
        r1 = await _dispatch_ad({"query": "帮我优化一下广告"})
        print(json.dumps(r1, ensure_ascii=False, indent=2, default=str))
        assert r1["success"] is False
        assert "缺少投放数据" in (r1["error_msg"] or "")

        # 4.2 不支持平台
        print("\n--- 4.2 不支持的平台 ---")
        r2 = await _dispatch_ad(
            {"query": "优化广告", "platform": "weibo", "spend": 100, "sku": "SKU-BAG-001"}
        )
        print(json.dumps(r2, ensure_ascii=False, indent=2, default=str))
        assert r2["success"] is False
        assert "不支持" in (r2["error_msg"] or "")

        # 4.3 高 ROAS → scale_up（规则）
        print("\n--- 4.3 正常优化（高 ROAS，规则期望 scale_up）---")
        r3 = await _dispatch_ad(
            {
                "query": "优化 Meta 广告 SKU-BAG-001",
                "sku": "SKU-BAG-001",
                "platform": "facebook",
                "campaign_id": "cmp-demo-1",
                "spend": 100,
                "revenue": 300,  # ROAS=3 >= 2*1.2
                "clicks": 1000,
                "conversions": 40,
                "daily_budget": 80,
                "bid": 1.2,
                "target_roas": 2.0,
            }
        )
        print(json.dumps(r3, ensure_ascii=False, indent=2, default=str))
        assert r3["success"] is True
        assert r3["reply_from"] == AGENT_EXEC
        data3 = r3["data"]
        assert data3.get("platform") == "meta"
        assert data3.get("sku") == "SKU-BAG-001"
        plan = (data3.get("ad_optimize") or {}).get("plan") or {}
        print("\n[优化计划 plan]")
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("\n[建议 suggested]")
        print(json.dumps((data3.get("ad_optimize") or {}).get("suggested"), ensure_ascii=False, indent=2))
        assert plan.get("action") == "scale_up"
        assert (data3.get("ad_optimize") or {}).get("source") == "rules"

        # 4.4 无转化高消耗 → scale_down + 可选利润测算
        print("\n--- 4.4 无转化高消耗 + profit_calc ---")
        r4 = await _dispatch_ad(
            {
                "sku": "SKU-DRESS-002",
                "platform": "tiktok",
                "spend": 80,
                "revenue": 0,
                "clicks": 200,
                "conversions": 0,
                "target_roas": 2.0,
                "cost": 22.5,
                "shipping": 9.2,
                "commission_rate": 0.078,
                "sell_price": 64.99,
            }
        )
        print(json.dumps(r4, ensure_ascii=False, indent=2, default=str))
        assert r4["success"] is True
        plan4 = (r4["data"].get("ad_optimize") or {}).get("plan") or {}
        profit = r4["data"].get("profit") or {}
        print("\n[plan.action]", plan4.get("action"), "| reasoning:", plan4.get("reasoning"))
        print("[profit]", json.dumps(profit, ensure_ascii=False))
        assert plan4.get("action") == "scale_down"
        assert "gross_profit" in profit or "profit_ratio" in profit

        # 记忆：高 ROAS 且 action!=hold 时应尝试写入
        assert mock_mem.safe_save_memory.await_count >= 1
        print("\n记忆写入次数:", mock_mem.safe_save_memory.await_count)

    finally:
        settings.DEEPSEEK_API_KEY = old_key
        mem_patch.stop()
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

    print("\n端到端 OK")


async def main():
    test_extract_platform()
    test_parse_metrics_from_query()
    test_build_skill_params()
    await test_ad_agent_e2e()
    print("\n" + "=" * 60)
    print("ad_agent 测试流程结束")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
