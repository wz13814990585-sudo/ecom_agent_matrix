"""Phase 2C-2A Competitor parser / workflow 测试。"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.skill.base_skill import SkillResult
from ecom_agent_matrix.core.tasking import normalize_task_context
from ecom_agent_matrix.core.tasking.result import (
    PARTIAL_SUCCESS,
    SKILL_FAILED,
    WORKFLOW_TIMEOUT,
)
from ecom_agent_matrix.modules.agent_cluster.handlers.competitor import (
    handle_price_warn,
    run_competitor_workflow,
)
from ecom_agent_matrix.modules.parsers.competitor import parse_competitor_request


def _monitor_success(price: float = 50) -> SkillResult:
    return SkillResult(
        success=True,
        data={
            "compete_price": price,
            "current_price_offset": -2,
            "warn_threshold": -10,
            "is_trigger_warn": False,
            "warn_message": "",
        },
    )


def _memory(*hits):
    memory = AsyncMock()
    memory.recall.return_value = list(hits)
    return memory


def test_competitor_canonical_sku_wins_over_query():
    request = parse_competitor_request(
        normalize_task_context(
            {"sku": "SKU-CANON", "query": "监控 Temu SKU-OTHER 价格"}
        )
    )
    assert request.sku == "SKU-CANON"


def test_explicit_competitor_wins_over_query():
    request = parse_competitor_request(
        normalize_task_context(
            {"query": "监控 Temu 价格", "competitor": "Amazon", "sku": "SKU-1"}
        )
    )
    assert request.competitor == "Amazon"


def test_explicit_compete_price_wins_over_query_price():
    request = parse_competitor_request(
        normalize_task_context(
            {
                "query": "Temu 竞品价 99",
                "sku": "SKU-1",
                "compete_price": 50,
            }
        )
    )
    assert request.compete_price == 50


def test_competitor_single_and_multi_modes():
    single = parse_competitor_request(
        normalize_task_context({"sku": "SKU-1", "competitor": "Temu"})
    )
    multi = parse_competitor_request(
        normalize_task_context({"sku": "SKU-1", "query": "多平台比价"})
    )
    assert single.mode == "single"
    assert multi.mode == "multi"


def test_multi_platform_prices_are_fetched_concurrently():
    entered: list[str] = []

    async def concurrent_skill(skill_name: str, params: dict):
        entered.append(params["competitor"])
        if len(entered) == 1:
            for _ in range(20):
                if len(entered) == 2:
                    break
                await asyncio.sleep(0)
        assert len(entered) == 2
        return SkillResult(
            success=True,
            data={"compete_price": 50 + len(entered), "price_source": "test"},
        )

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.competitor.exec_skill",
            side_effect=concurrent_skill,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.competitor.llm_explain",
            new=AsyncMock(return_value=("advice", "template", "")),
        ):
            return await run_competitor_workflow(
                {
                    "sku": "SKU-1",
                    "multi_compare": True,
                    "platforms": ["Temu", "Amazon"],
                }
            )

    result = asyncio.run(scenario())
    assert result.success is True
    assert [row["competitor"] for row in result.data["comparisons"]] == ["Temu", "Amazon"]


def test_multi_platform_partial_failure_preserves_successful_rows():
    async def mixed(skill_name: str, params: dict):
        if params["competitor"] == "Temu":
            return SkillResult(success=False, error_code="TIMEOUT", error_msg="timeout")
        return SkillResult(success=True, data={"compete_price": 60})

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.competitor.exec_skill",
            side_effect=mixed,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.competitor.llm_explain",
            new=AsyncMock(return_value=("advice", "template", "")),
        ):
            return await run_competitor_workflow(
                {"sku": "SKU-1", "multi_compare": True, "platforms": ["Temu", "Amazon"]}
            )

    result = asyncio.run(scenario())
    assert result.success is True
    assert result.partial_success is True
    assert result.error_code == PARTIAL_SUCCESS
    assert result.data["comparisons"][1]["compete_price"] == 60
    assert result.metadata["skill_error_codes"]["Temu"] == "TIMEOUT"


def test_multi_platform_all_fail_is_structured_failure():
    async def failed(skill_name: str, params: dict):
        return SkillResult(success=False, error_code="EXECUTION_ERROR", error_msg="failed")

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.competitor.exec_skill",
            side_effect=failed,
        ):
            return await run_competitor_workflow(
                {"sku": "SKU-1", "multi_compare": True, "platforms": ["Temu", "Amazon"]}
            )

    result = asyncio.run(scenario())
    assert result.success is False
    assert result.error_code == SKILL_FAILED
    assert len(result.data["comparisons"]) == 2


def test_multi_platform_workflow_deadline_is_structured():
    async def blocked(skill_name: str, params: dict):
        await asyncio.sleep(1)
        return SkillResult(success=True)

    async def scenario():
        with patch.object(settings, "QUERY_SKILL_TIMEOUT", 0.01), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.competitor.exec_skill",
            side_effect=blocked,
        ):
            return await run_competitor_workflow(
                {"sku": "SKU-1", "multi_compare": True, "platforms": ["Temu", "Amazon"]}
            )

    result = asyncio.run(scenario())
    assert result.success is False
    assert result.error_code == WORKFLOW_TIMEOUT


def test_price_monitor_skill_error_code_is_preserved():
    failure = SkillResult(
        success=False,
        error_code="PERMISSION_DENIED",
        error_msg="denied",
    )

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.competitor.exec_skill",
            new=AsyncMock(return_value=failure),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.competitor._mem",
            return_value=_memory(),
        ):
            return await run_competitor_workflow(
                {
                    "sku": "SKU-1",
                    "competitor": "Temu",
                    "compete_price": 50,
                }
            )

    result = asyncio.run(scenario())
    assert result.error_code == SKILL_FAILED
    assert result.metadata["skill_error_code"] == "PERMISSION_DENIED"


def test_long_memory_price_is_never_used_as_current_price():
    calls: list[tuple[str, dict]] = []

    async def skills(skill_name: str, params: dict):
        calls.append((skill_name, params))
        if skill_name == "competitor_price":
            return SkillResult(success=True, data={"compete_price": 50})
        return _monitor_success(50)

    async def scenario():
        old_price = {"meta": {"compete_price": 1}, "content": "old price 1"}
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.competitor.exec_skill",
            side_effect=skills,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.competitor._mem",
            return_value=_memory(old_price),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.competitor.llm_explain",
            new=AsyncMock(return_value=("advice", "template", "")),
        ):
            return await run_competitor_workflow({"sku": "SKU-1", "competitor": "Temu"})

    result = asyncio.run(scenario())
    assert result.data["compete_price"] == 50
    assert [name for name, _ in calls] == ["competitor_price", "price_monitor"]
    assert calls[1][1]["compete_price"] == 50


def test_competitor_legacy_tuple_compatibility():
    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.competitor.exec_skill",
            new=AsyncMock(return_value=_monitor_success()),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.competitor._mem",
            return_value=_memory(),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.competitor.llm_explain",
            new=AsyncMock(return_value=("advice", "template", "")),
        ):
            return await handle_price_warn(
                {"sku": "SKU-1", "competitor": "Temu", "compete_price": 50}
            )

    ok, error, data = asyncio.run(scenario())
    assert ok is True
    assert error == ""
    assert data["_workflow"]["metadata"]["workflow"] == "competitor"
