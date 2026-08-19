"""API 鉴权与路由冒烟（mock 下发）。"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.result_waiter import GatewayResultWaiter


def _client():
    # 同时 patch 注册表与 main 引用，避免 TestClient lifespan 真拉起 Agent 循环
    with patch("ecom_agent_matrix.core.mcp.registry.start_all_agents", new=AsyncMock()), patch(
        "ecom_agent_matrix.api.main.start_all_agents", new=AsyncMock()
    ):
        from ecom_agent_matrix.api import main as main_mod

        return TestClient(main_mod.app)


def test_health():
    with _client() as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["agents_count"] >= 1
        assert "api_auth_enabled" in body


def test_list_agents_requires_auth_when_key_set():
    with patch.object(settings, "API_KEY", "secret-test"), _client() as client:
        r = client.get("/api/v1/agents")
        assert r.status_code == 401
        r2 = client.get("/api/v1/agents", headers={"X-API-Key": "secret-test"})
        assert r2.status_code == 200
        assert "master_planning" in r2.json()["agents"]


def test_health_ready_mocked():
    fake = {
        "ready": True,
        "postgres": {"ok": True},
        "redis": {"ok": True},
    }
    with patch(
        "ecom_agent_matrix.api.main.readiness_report",
        new=AsyncMock(return_value=fake),
    ), _client() as client:
        r = client.get("/health/ready")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"


def test_health_ready_unavailable():
    fake = {
        "ready": False,
        "postgres": {"ok": False, "error": "down"},
        "redis": {"ok": True},
    }
    with patch(
        "ecom_agent_matrix.api.main.readiness_report",
        new=AsyncMock(return_value=fake),
    ), _client() as client:
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["status"] == "not_ready"


def test_auth_required_when_api_key_set():
    with patch.object(settings, "API_KEY", "secret-test"), _client() as client:
        r = client.post("/api/v1/tasks", json={"query": "帮我写社媒文案"})
        assert r.status_code == 401


def test_create_task_mocked():
    fake = {
        "task_id": "t1",
        "target": "master_planning",
        "reply_from": "master_planning",
        "success": True,
        "data": {"mode": "react"},
        "error_msg": "",
        "msg_type": "master_task_result",
        "summary": "已规划社媒任务",
    }
    with patch.object(settings, "API_KEY", ""), patch(
        "ecom_agent_matrix.api.route_task.dispatch_to_master",
        new=AsyncMock(return_value=fake),
    ), _client() as client:
        r = client.post(
            "/api/v1/tasks",
            json={"query": "为「防水背包」生成tiktok文案", "task_type": "social_marketing"},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True
        assert r.json()["task_id"] == "t1"
        assert r.json()["summary"]


def test_customer_chat_mocked():
    fake = {
        "task_id": "c1",
        "target": "master_planning",
        "reply_from": "master_planning",
        "success": True,
        "data": {"answer": "您好"},
        "error_msg": "",
        "msg_type": "agent_reply",
        "summary": "客服已回复",
    }
    with patch(
        "ecom_agent_matrix.api.route_customer.dispatch_to_master",
        new=AsyncMock(return_value=fake),
    ), _client() as client:
        r = client.post("/api/v1/customer/chat", json={"query": "我要退款", "lang": "zh"})
        assert r.status_code == 200
        assert r.json()["data"]["answer"] == "您好"


def test_warn_competitor_mocked():
    fake = {
        "task_id": "w1",
        "target": "data_query",
        "reply_from": "data_query",
        "success": True,
        "data": {"is_trigger_warn": False, "compete_price": 29.9, "advice": "可继续观察"},
        "error_msg": "",
        "msg_type": "agent_reply",
        "summary": "竞品价稳定",
    }
    with patch.object(settings, "API_KEY", ""), patch(
        "ecom_agent_matrix.api.route_warn.dispatch_and_wait",
        new=AsyncMock(return_value=fake),
    ), _client() as client:
        r = client.post(
            "/api/v1/warn/competitor",
            json={
                "sku": "SKU-BAG-001",
                "competitor": "Temu",
                "compete_price": 29.9,
                "via_master": False,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["data"]["compete_price"] == 29.9
        assert body["summary"]


def test_gateway_waiter_ignores_non_api_target():
    async def _run():
        GatewayResultWaiter.begin("tid-1")
        sub = MCPMessage(
            task_id="tid-1",
            sender="stock_agent",
            target="master_planning",
            priority=3,
            content={"type": "agent_reply", "ref_task_id": "tid-1", "success": True, "data": {}},
        )
        assert GatewayResultWaiter.submit(sub) is False
        final = MCPMessage(
            task_id="tid-1",
            sender="master_planning",
            target=settings.API_SENDER,
            priority=3,
            content={
                "type": "master_task_result",
                "ref_task_id": "tid-1",
                "success": True,
                "data": {"ok": 1},
            },
        )
        assert GatewayResultWaiter.submit(final) is True
        reply = await GatewayResultWaiter.wait("tid-1", 1.0)
        assert reply is not None and reply.content["data"]["ok"] == 1

    asyncio.run(_run())


if __name__ == "__main__":
    test_gateway_waiter_ignores_non_api_target()
    test_health()
    test_list_agents_requires_auth_when_key_set()
    test_health_ready_mocked()
    test_health_ready_unavailable()
    test_auth_required_when_api_key_set()
    test_create_task_mocked()
    test_customer_chat_mocked()
    test_warn_competitor_mocked()
    print("✅ API 测试通过")
