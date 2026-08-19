"""Phase 2C-2B Report parser / workflow tests。"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from ecom_agent_matrix.core.skill.base_skill import SkillResult
from ecom_agent_matrix.core.tasking import normalize_task_context
from ecom_agent_matrix.core.tasking.result import SKILL_FAILED, UNSUPPORTED_REPORT_TYPE
from ecom_agent_matrix.modules.agent_cluster.handlers.report import handle_report, run_report_workflow
from ecom_agent_matrix.modules.parsers.report import parse_report_request


def test_report_explicit_type_and_days_win_query():
    request = parse_report_request(normalize_task_context({
        "query": "最近30天销售报表", "report_type": "stock", "days": 7,
    }))
    assert request.report_type == "stock"
    assert request.days == 7


def test_report_days_parsed_from_query():
    assert parse_report_request(normalize_task_context({"query": "最近30天销售报表"})).days == 30


@pytest.mark.parametrize("payload", [{"days": 0}, {"days": 91}, {"top_k": 0}, {"top_k": 21}])
def test_report_numeric_validation(payload):
    with pytest.raises(ValidationError):
        parse_report_request(normalize_task_context(payload))


def test_report_unsupported_type_is_structured():
    result = asyncio.run(run_report_workflow({"report_type": "unknown"}))
    assert result.error_code == UNSUPPORTED_REPORT_TYPE


def test_ops_report_skill_error_code_preserved():
    failure = SkillResult(success=False, error_code="TIMEOUT", error_msg="timeout")
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.report.exec_skill", new=AsyncMock(return_value=failure)):
            return await run_report_workflow({"report_type": "sales"})
    result = asyncio.run(scenario())
    assert result.error_code == SKILL_FAILED
    assert result.metadata["skill_error_code"] == "TIMEOUT"


def test_report_legacy_tuple_compatibility():
    success = SkillResult(success=True, data={"report_type": "sales", "summary": "ok"})
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.report.exec_skill", new=AsyncMock(return_value=success)):
            return await handle_report({"report_type": "sales"})
    ok, _, data = asyncio.run(scenario())
    assert ok and data["_workflow"]["metadata"]["workflow"] == "report"
