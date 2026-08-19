"""Phase 2C-2B Ad parser / workflow tests。"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.core.skill.base_skill import SkillResult
from ecom_agent_matrix.core.tasking import normalize_task_context
from ecom_agent_matrix.core.tasking.result import INVALID_REQUEST, PARTIAL_SUCCESS, SKILL_FAILED, UNSUPPORTED_PLATFORM
from ecom_agent_matrix.modules.agent_cluster.handlers.ad import handle_ad, run_ad_workflow
from ecom_agent_matrix.modules.parsers.ad import parse_ad_request


def _ad_success(action="hold"):
    return SkillResult(success=True, data={"plan": {"action": action, "bid_adjust_pct": 5, "budget_adjust_pct": 10}})


def _memory():
    memory = AsyncMock()
    memory.recall.return_value = []
    memory.safe_save_memory.return_value = 1
    return memory


def test_ad_explicit_platform_and_metrics_win_query():
    request = parse_ad_request(normalize_task_context({
        "query": "TikTok spend 20 clicks 2", "platform": "google", "spend": 100, "clicks": 9,
    }))
    assert request.platform == "google"
    assert request.spend == 100
    assert request.clicks == 9


def test_ad_unsupported_platform_is_structured():
    result = asyncio.run(run_ad_workflow({"platform": "pinterest", "spend": 10}))
    assert result.error_code == UNSUPPORTED_PLATFORM


def test_ad_numeric_validation():
    result = asyncio.run(run_ad_workflow({"spend": -1}))
    assert result.error_code == INVALID_REQUEST


def test_ad_partial_profit_inputs_are_rejected():
    result = asyncio.run(run_ad_workflow({"spend": 10, "cost": 2, "shipping": 1}))
    assert result.error_code == INVALID_REQUEST
    assert set(result.data["missing_profit_fields"]) == {"commission_rate", "sell_price"}


def test_ad_complete_profit_inputs_call_profit_calc():
    calls = []
    async def skills(name, params):
        calls.append((name, params))
        return _ad_success() if name == "ad_optimize" else SkillResult(success=True, data={"gross_profit": 5})
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.ad.exec_skill", side_effect=skills):
            return await run_ad_workflow({
                "spend": 10, "cost": 2, "shipping": 1, "commission_rate": .1, "sell_price": 10,
            })
    result = asyncio.run(scenario())
    assert result.success is True
    assert calls[1] == ("profit_calc", {"cost": 2.0, "shipping": 1.0, "commission_rate": .1, "sell_price": 10.0})


def test_ad_success_profit_failure_is_partial():
    async def skills(name, params):
        return _ad_success() if name == "ad_optimize" else SkillResult(success=False, error_code="TIMEOUT", error_msg="timeout")
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.ad.exec_skill", side_effect=skills):
            return await run_ad_workflow({
                "spend": 10, "cost": 2, "shipping": 1, "commission_rate": .1, "sell_price": 10,
            })
    result = asyncio.run(scenario())
    assert result.success and result.partial_success
    assert result.error_code == PARTIAL_SUCCESS
    assert result.metadata["skill_error_codes"]["profit_calc"] == "TIMEOUT"


def test_ad_skill_error_code_preserved():
    failure = SkillResult(success=False, error_code="EXECUTION_ERROR", error_msg="failed")
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.ad.exec_skill", new=AsyncMock(return_value=failure)):
            return await run_ad_workflow({"spend": 10})
    result = asyncio.run(scenario())
    assert result.error_code == SKILL_FAILED
    assert result.metadata["skill_error_code"] == "EXECUTION_ERROR"


def test_ad_memory_recall_failure_does_not_fail_workflow():
    memory = _memory()
    memory.recall.side_effect = RuntimeError("down")
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.ad._mem", return_value=memory), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.ad.exec_skill", new=AsyncMock(return_value=_ad_success())
        ):
            return await run_ad_workflow({"sku": "SKU-1"})
    result = asyncio.run(scenario())
    assert result.success is True
    assert result.metadata["memory_errors"]


def test_ad_memory_save_failure_does_not_fail_workflow():
    memory = _memory()
    memory.safe_save_memory.side_effect = RuntimeError("down")
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.ad._mem", return_value=memory), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.ad.exec_skill", new=AsyncMock(return_value=_ad_success("scale_up"))
        ):
            return await run_ad_workflow({"sku": "SKU-1"})
    result = asyncio.run(scenario())
    assert result.success is True
    assert result.error_code == PARTIAL_SUCCESS


def test_ad_legacy_tuple_compatibility():
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.ad.exec_skill", new=AsyncMock(return_value=_ad_success())):
            return await handle_ad({"spend": 10})
    ok, _, data = asyncio.run(scenario())
    assert ok and data["_workflow"]["metadata"]["workflow"] == "ad"
