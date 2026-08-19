from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.api import main as main_module
from ecom_agent_matrix.api.health import readiness_report


def test_shutdown_cancels_agent_and_closes_all_owned_clients():
    async def scenario():
        agent = asyncio.create_task(asyncio.Event().wait())
        with patch.object(main_module, "cancel_master_tasks", new=AsyncMock()) as master, patch.object(
            main_module, "close_http_session", new=AsyncMock()
        ) as llm, patch.object(
            main_module.AsyncRedisClient, "close", new=AsyncMock()
        ) as redis, patch.object(
            main_module.AsyncPGClient, "close", new=AsyncMock()
        ) as postgres:
            await main_module.shutdown_runtime(agent)
        return agent, master, llm, redis, postgres

    agent, master, llm, redis, postgres = asyncio.run(scenario())
    assert agent.cancelled()
    master.assert_awaited_once()
    llm.assert_awaited_once()
    redis.assert_awaited_once()
    postgres.assert_awaited_once()


def test_readiness_reports_dependencies_and_llm_degraded_is_optional():
    async def scenario():
        with patch("ecom_agent_matrix.api.health.check_postgres", new=AsyncMock(return_value={"ok": True, "status": "ok"})), patch(
            "ecom_agent_matrix.api.health.check_redis", new=AsyncMock(return_value={"ok": True, "status": "ok"})
        ), patch("ecom_agent_matrix.core.llm.router.is_llm_configured", return_value=False), patch(
            "ecom_agent_matrix.api.health.settings.LLM_REQUIRED_FOR_READINESS", False
        ):
            optional = await readiness_report(agents_alive=True)
        with patch("ecom_agent_matrix.api.health.check_postgres", new=AsyncMock(return_value={"ok": True, "status": "ok"})), patch(
            "ecom_agent_matrix.api.health.check_redis", new=AsyncMock(return_value={"ok": True, "status": "ok"})
        ), patch("ecom_agent_matrix.core.llm.router.is_llm_configured", return_value=False), patch(
            "ecom_agent_matrix.api.health.settings.LLM_REQUIRED_FOR_READINESS", True
        ):
            required = await readiness_report(agents_alive=True)
        return optional, required

    optional, required = asyncio.run(scenario())
    assert optional["ready"] and optional["degraded"]
    assert optional["dependencies"]["postgres"] == "ok"
    assert not required["ready"]
