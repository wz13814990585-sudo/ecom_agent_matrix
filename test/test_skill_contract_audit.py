"""Phase 2D：Skill contract、依赖边界与风险原子化审计。"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import ecom_agent_matrix.modules.skills  # noqa: F401
from ecom_agent_matrix.config.constants import AGENT_EXEC, AGENT_QUERY
from ecom_agent_matrix.core.skill.base_skill import SkillResult
from ecom_agent_matrix.core.skill.skill_registry import (
    exec_skill,
    skill_container,
    skill_execution_context,
)
from ecom_agent_matrix.core.tasking.result import PARTIAL_SUCCESS
from ecom_agent_matrix.modules.agent_cluster.handlers.risk import (
    handle_risk,
    run_risk_workflow,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SKILLS = [
    (name, skill_cls)
    for name, skill_cls in sorted(skill_container.items())
    if skill_cls.__module__.startswith("ecom_agent_matrix.modules.skills.")
]


@pytest.mark.parametrize("skill_name,skill_cls", PRODUCTION_SKILLS)
def test_registered_skill_contract_and_metadata(skill_name, skill_cls):
    spec = skill_cls.spec()
    for field in (
        "read_only",
        "side_effect",
        "risk_level",
        "timeout_seconds",
        "idempotent",
        "input_model",
        "output_model",
    ):
        assert field in skill_cls.__dict__, f"{skill_name} must explicitly declare {field}"
    assert skill_name == spec.name and spec.name.strip()
    assert spec.input_model is not None
    assert spec.output_model is not None
    assert spec.input_model.model_config.get("extra") == "forbid"
    assert spec.output_model.model_config.get("extra") == "forbid"
    assert spec.risk_level in {"low", "medium", "high", "critical"}
    assert spec.timeout_seconds > 0
    assert not (spec.read_only and spec.side_effect)
    assert not (spec.side_effect and spec.read_only)
    if spec.side_effect:
        assert spec.idempotent is False
    if spec.deprecated:
        assert spec.replacement


def test_all_production_skills_are_covered_by_audit():
    assert len(PRODUCTION_SKILLS) == 19


def test_parser_and_core_tasking_dependency_boundaries():
    parser_dir = PROJECT_ROOT / "ecom_agent_matrix" / "modules" / "parsers"
    for path in parser_dir.glob("*.py"):
        assert "ecom_agent_matrix.modules.skills" not in path.read_text(encoding="utf-8")

    tasking_dir = PROJECT_ROOT / "ecom_agent_matrix" / "core" / "tasking"
    for path in tasking_dir.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "ecom_agent_matrix.modules.skills" not in source
        assert "ecom_agent_matrix.modules.agent_cluster" not in source


@pytest.mark.parametrize(
    "amount,count,is_risk,tags",
    [
        (500, 20, False, []),
        (500.01, 1, True, ["大额订单"]),
        (10, 21, True, ["批量囤货"]),
        (501, 21, True, ["大额订单", "批量囤货"]),
    ],
)
def test_evaluate_order_risk_is_deterministic_and_never_writes(
    amount, count, is_risk, tags
):
    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.skills.risk_control.AsyncPGClient.execute_sql",
            new=AsyncMock(),
        ) as execute:
            result = await exec_skill(
                "evaluate_order_risk",
                {"order_no": "ORD-1", "total_amount": amount, "buy_count": count},
            )
        return result, execute

    result, execute = asyncio.run(scenario())
    assert result.success is True
    assert result.data["is_risk"] is is_risk
    assert result.data["risk_tags"] == tags
    execute.assert_not_awaited()


def test_record_order_risk_permission_matrix_and_single_insert():
    params = {
        "order_no": "ORD-1",
        "risk_type": "order_abnormal",
        "risk_desc": "大额订单",
    }

    async def scenario():
        no_context = await exec_skill("record_order_risk", params)
        with skill_execution_context(AGENT_QUERY):
            query = await exec_skill("record_order_risk", params)
        with patch(
            "ecom_agent_matrix.modules.skills.risk_control.AsyncPGClient.execute_sql",
            new=AsyncMock(return_value=[[7]]),
        ) as execute:
            with skill_execution_context(AGENT_EXEC):
                allowed = await exec_skill("record_order_risk", params)
        return no_context, query, allowed, execute

    no_context, query, allowed, execute = asyncio.run(scenario())
    assert no_context.error_code == "PERMISSION_DENIED"
    assert query.error_code == "PERMISSION_DENIED"
    assert allowed.success is True and allowed.data["record_id"] == 7
    assert allowed.metadata["agent_id"] == AGENT_EXEC
    execute.assert_awaited_once()
    sql, sql_params = execute.await_args.args
    assert "INSERT INTO risk_record" in sql
    assert sql_params == ["ORD-1", "order_abnormal", "大额订单"]
    spec = skill_container["record_order_risk"].spec()
    assert spec.risk_level == "high" and spec.side_effect and not spec.read_only


def test_risk_workflow_skips_record_for_safe_order():
    evaluation = SkillResult(
        success=True,
        data={"is_risk": False, "risk_tags": [], "risk_detail": "无异常风险"},
    )

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.risk.exec_skill",
            new=AsyncMock(return_value=evaluation),
        ) as execute:
            result = await run_risk_workflow(
                {"order_no": "ORD-1", "total_amount": 10, "buy_count": 1}
            )
        return result, execute

    result, execute = asyncio.run(scenario())
    assert result.success and result.data["record"]["skipped"]
    execute.assert_awaited_once()


def test_risk_workflow_records_risk_and_handles_partial_failure():
    evaluation = SkillResult(
        success=True,
        data={"is_risk": True, "risk_tags": ["大额订单"], "risk_detail": "大额订单"},
    )
    failed_record = SkillResult(
        success=False, error_code="SKILL_FAILED", error_msg="record failed"
    )

    async def scenario():
        execute = AsyncMock(side_effect=[evaluation, failed_record])
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.risk.exec_skill",
            new=execute,
        ):
            result = await run_risk_workflow(
                {"order_no": "ORD-1", "total_amount": 501, "buy_count": 1}
            )
        return result, execute

    result, execute = asyncio.run(scenario())
    assert result.success is True and result.partial_success is True
    assert result.error_code == PARTIAL_SUCCESS
    assert result.data["risk"]["data"]["is_risk"] is True
    assert execute.await_args_list[1].args == (
        "record_order_risk",
        {"order_no": "ORD-1", "risk_type": "order_abnormal", "risk_desc": "大额订单"},
    )


def test_risk_legacy_tuple_and_deprecated_skill_compatibility():
    async def scenario():
        with skill_execution_context(AGENT_EXEC):
            with patch(
                "ecom_agent_matrix.modules.skills.risk_control.AsyncPGClient.execute_sql",
                new=AsyncMock(return_value=[[9]]),
            ):
                legacy_skill = await exec_skill(
                    "order_risk_check",
                    {"order_no": "ORD-1", "total_amount": 501, "buy_count": 1},
                )
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.risk.exec_skill",
            new=AsyncMock(
                return_value=SkillResult(
                    success=True,
                    data={"is_risk": False, "risk_tags": [], "risk_detail": "无异常风险"},
                )
            ),
        ):
            legacy_tuple = await handle_risk(
                {"order_no": "ORD-2", "total_amount": 10, "buy_count": 1}
            )
        return legacy_skill, legacy_tuple

    legacy_skill, (ok, _error, data) = asyncio.run(scenario())
    assert legacy_skill.success and legacy_skill.data["is_risk"]
    assert legacy_skill.metadata["deprecated"] is True
    assert legacy_skill.metadata["replacement"] == "evaluate_order_risk + record_order_risk"
    assert ok and data["_workflow"]["metadata"]["workflow"] == "risk"


def test_skill_executor_fills_blank_failure_error_code():
    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.skills.ai_prompt_gen.AIPromptGenTool.run",
            new=AsyncMock(return_value=SkillResult(success=False, error_msg="failed")),
        ):
            return await exec_skill("ai_prompt_generate", {"product": "bag"})

    result = asyncio.run(scenario())
    assert result.success is False
    assert result.error_code == "SKILL_FAILED"


def test_contract_validation_does_not_echo_secret_input():
    secret = "real-api-token-must-not-leak"
    result = asyncio.run(
        exec_skill(
            "profit_calc",
            {
                "cost": 1,
                "shipping": 1,
                "commission_rate": 0.1,
                "sell_price": 10,
                "api_token": secret,
            },
        )
    )
    assert result.error_code == "VALIDATION_ERROR"
    assert secret not in result.error_msg
