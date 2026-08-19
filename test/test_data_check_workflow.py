"""Phase 2C-2A DataCheck parser / workflow 测试。"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from ecom_agent_matrix.core.skill.base_skill import SkillResult
from ecom_agent_matrix.core.tasking import normalize_task_context
from ecom_agent_matrix.core.tasking.result import PARTIAL_SUCCESS, SKILL_FAILED
from ecom_agent_matrix.modules.agent_cluster.handlers.data_check import (
    handle_data_check,
    run_data_check_workflow,
)
from ecom_agent_matrix.modules.parsers import data_check as data_parser
from ecom_agent_matrix.modules.parsers.data_check import parse_data_check_request


def _integrity_success(*, passed: bool = True) -> SkillResult:
    return SkillResult(
        success=True,
        data={"passed": passed, "issue_count": 0 if passed else 2, "issues": []},
    )


def test_data_check_order_no_parsing():
    explicit = parse_data_check_request(
        normalize_task_context({"order_no": "ORD-EXPLICIT", "query": "订单 ORD-OTHER"})
    )
    inferred = parse_data_check_request(
        normalize_task_context({"query": "查询订单 ORD-2026-ABC"})
    )
    assert explicit.order_no == "ORD-EXPLICIT"
    assert inferred.order_no == "ORD-2026-ABC"


@pytest.mark.parametrize(
    ("query", "scope"),
    [("检查订单数据", "order"), ("检查商品主数据", "goods"), ("数据完整性检查", "full")],
)
def test_data_check_scope_parsing(query: str, scope: str):
    assert parse_data_check_request(normalize_task_context({"query": query})).scope == scope


def test_data_check_explicit_scope_wins_over_query():
    request = parse_data_check_request(
        normalize_task_context({"query": "检查订单数据", "scope": "goods"})
    )
    assert request.scope == "goods"


@pytest.mark.parametrize("limit", [0, 501])
def test_data_check_limit_validation(limit: int):
    with pytest.raises(ValidationError):
        parse_data_check_request(normalize_task_context({"limit": limit}))


def test_custom_sql_always_goes_through_safe_sql_skill():
    calls: list[tuple[str, dict]] = []

    async def skills(skill_name: str, params: dict):
        calls.append((skill_name, params))
        if skill_name == "data_integrity_check":
            return _integrity_success()
        return SkillResult(success=True, data={"query_result": [[1]]})

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.data_check.exec_skill",
            side_effect=skills,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.data_check.llm_explain",
            new=AsyncMock(return_value=("summary", "template", "")),
        ):
            return await run_data_check_workflow(
                {"custom_sql": "SELECT 1", "sql_params": [7]}
            )

    result = asyncio.run(scenario())
    assert result.success is True
    assert calls[1] == ("safe_sql_query", {"params": [7], "sql": "SELECT 1"})


def test_natural_language_db_query_goes_through_safe_sql_skill():
    calls: list[tuple[str, dict]] = []

    async def skills(skill_name: str, params: dict):
        calls.append((skill_name, params))
        return _integrity_success() if skill_name == "data_integrity_check" else SkillResult(success=True)

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.data_check.exec_skill",
            side_effect=skills,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.data_check.llm_explain",
            new=AsyncMock(return_value=("summary", "template", "")),
        ):
            return await run_data_check_workflow({"query": "有多少订单"})

    asyncio.run(scenario())
    assert calls[1] == ("safe_sql_query", {"params": [], "query": "有多少订单"})


def test_data_check_parser_has_no_db_or_skill_access():
    source = inspect.getsource(data_parser)
    assert "AsyncPGClient" not in source
    assert "exec_skill" not in source
    assert "await " not in source


def test_integrity_success_and_sql_failure_is_partial_success():
    async def skills(skill_name: str, params: dict):
        if skill_name == "data_integrity_check":
            return _integrity_success()
        return SkillResult(
            success=False,
            error_code="VALIDATION_ERROR",
            error_msg="unsafe sql",
        )

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.data_check.exec_skill",
            side_effect=skills,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.data_check.llm_explain",
            new=AsyncMock(return_value=("summary", "template", "")),
        ):
            return await run_data_check_workflow({"custom_sql": "DELETE FROM goods"})

    result = asyncio.run(scenario())
    assert result.success is True
    assert result.partial_success is True
    assert result.error_code == PARTIAL_SUCCESS
    assert result.metadata["skill_error_codes"]["safe_sql_query"] == "VALIDATION_ERROR"


def test_integrity_skill_error_code_is_preserved():
    failure = SkillResult(
        success=False,
        error_code="TIMEOUT",
        error_msg="timeout",
    )

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.data_check.exec_skill",
            new=AsyncMock(return_value=failure),
        ):
            return await run_data_check_workflow({"query": "数据完整性检查"})

    result = asyncio.run(scenario())
    assert result.success is False
    assert result.error_code == SKILL_FAILED
    assert result.metadata["skill_error_code"] == "TIMEOUT"


def test_data_check_legacy_tuple_compatibility():
    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.data_check.exec_skill",
            new=AsyncMock(return_value=_integrity_success()),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.data_check.llm_explain",
            new=AsyncMock(return_value=("summary", "template", "")),
        ):
            return await handle_data_check(normalize_task_context({"scope": "full"}))

    ok, error, data = asyncio.run(scenario())
    assert ok is True
    assert error == ""
    assert data["_workflow"]["error_code"] == ""
    assert data["_workflow"]["metadata"]["workflow"] == "data_check"
