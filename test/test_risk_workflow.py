"""Phase 2C-2B Risk parser / workflow tests。"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from ecom_agent_matrix.config.constants import AGENT_EXEC, AGENT_QUERY
from ecom_agent_matrix.core.skill.base_skill import SkillResult
from ecom_agent_matrix.core.skill.skill_registry import skill_execution_context
from ecom_agent_matrix.core.tasking import normalize_task_context
from ecom_agent_matrix.modules.agent_cluster.handlers import data_check
from ecom_agent_matrix.modules.agent_cluster.handlers.risk import handle_risk, run_risk_workflow
from ecom_agent_matrix.modules.parsers.risk import parse_risk_request


def test_risk_canonical_order_no_wins_query():
    request = parse_risk_request(normalize_task_context({
        "order_no": "ORD-CANON", "query": "check ORD-OTHER", "total_amount": 10, "buy_count": 1,
    }))
    assert request.order_no == "ORD-CANON"


def test_risk_order_no_extracted_from_query():
    request = parse_risk_request(normalize_task_context({
        "query": "check ORD-2026-ABC", "total_amount": 10, "buy_count": 1,
    }))
    assert request.order_no == "ORD-2026-ABC"


@pytest.mark.parametrize("payload", [
    {"order_no": "ORD-1", "total_amount": -1, "buy_count": 1},
    {"order_no": "ORD-1", "total_amount": 1, "buy_count": 0},
])
def test_risk_amount_and_count_validation(payload):
    with pytest.raises(ValidationError):
        parse_risk_request(normalize_task_context(payload))


def test_risk_buy_num_legacy_alias():
    request = parse_risk_request(normalize_task_context({
        "order_no": "ORD-1", "total_amount": 10, "buy_num": 3,
    }))
    assert request.buy_count == 3


def test_risk_workflow_calls_evaluate_risk_skill():
    success = SkillResult(success=True, data={"is_risk": False})
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.risk.exec_skill", new=AsyncMock(return_value=success)) as execute:
            result = await run_risk_workflow({"order_no": "ORD-1", "total_amount": 10, "buy_count": 1})
        return result, execute.await_args.args
    result, args = asyncio.run(scenario())
    assert result.success is True
    assert args == ("evaluate_order_risk", {"order_no": "ORD-1", "total_amount": 10.0, "buy_count": 1})


def test_query_context_can_evaluate_non_risk_order_without_write():
    async def scenario():
        with skill_execution_context(AGENT_QUERY):
            return await run_risk_workflow({"order_no": "ORD-1", "total_amount": 10, "buy_count": 1})
    result = asyncio.run(scenario())
    assert result.success is True
    assert result.data["record"]["skipped"] is True


def test_exec_context_can_execute_risk_write_skill():
    async def scenario():
        with skill_execution_context(AGENT_EXEC):
            return await run_risk_workflow({"order_no": "ORD-1", "total_amount": 10, "buy_count": 1})
    result = asyncio.run(scenario())
    assert result.success is True
    assert result.data["risk"]["data"]["is_risk"] is False


def test_data_check_no_longer_contains_risk_handler():
    source = inspect.getsource(data_check)
    assert "def handle_risk" not in source
    assert "order_risk_check" not in source


def test_risk_legacy_tuple_compatibility():
    success = SkillResult(success=True, data={"is_risk": False})
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.risk.exec_skill", new=AsyncMock(return_value=success)):
            return await handle_risk({"order_no": "ORD-1", "total_amount": 10, "buy_count": 1})
    ok, _, data = asyncio.run(scenario())
    assert ok and data["_workflow"]["metadata"]["workflow"] == "risk"


def test_risky_order_surfaces_approval_and_does_not_retry_write():
    evaluation = SkillResult(
        success=True,
        data={"is_risk": True, "risk_tags": ["大额订单"], "risk_detail": "大额订单"},
    )
    pending = SkillResult(
        success=False,
        error_code="APPROVAL_REQUIRED",
        error_msg="approval required",
        data={"approval_required": True, "approval_id": "approval-1"},
    )

    async def scenario():
        execute = AsyncMock(side_effect=[evaluation, pending])
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.risk.exec_skill", new=execute
        ):
            result = await run_risk_workflow(
                {"order_no": "ORD-1", "total_amount": 501, "buy_count": 1}
            )
        return result, execute

    result, execute = asyncio.run(scenario())
    assert result.success and result.partial_success
    assert result.data["approval_required"] is True
    assert result.data["approval_id"] == "approval-1"
    assert execute.await_count == 2


def test_approved_risky_order_records_once_and_write_failure_is_not_retried():
    evaluation = SkillResult(
        success=True,
        data={"is_risk": True, "risk_tags": ["大额订单"], "risk_detail": "大额订单"},
    )

    async def run(record):
        execute = AsyncMock(side_effect=[evaluation, record])
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.risk.exec_skill", new=execute
        ):
            result = await run_risk_workflow(
                {"order_no": "ORD-1", "total_amount": 501, "buy_count": 1}
            )
        return result, execute

    success, success_exec = asyncio.run(run(SkillResult(success=True, data={"record_id": 7})))
    failure, failure_exec = asyncio.run(run(SkillResult(
        success=False, error_code="EXECUTION_ERROR", error_msg="write failed"
    )))
    assert success.success and not success.partial_success
    assert success.data["record"]["data"]["record_id"] == 7
    assert failure.success and failure.partial_success
    assert success_exec.await_count == 2 and failure_exec.await_count == 2
