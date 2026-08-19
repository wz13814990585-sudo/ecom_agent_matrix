"""Phase 2A：SkillSpec / SkillExecutor / Contract 回归测试。"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel, ConfigDict

import ecom_agent_matrix.modules.skills  # noqa: F401
from ecom_agent_matrix.config.constants import AGENT_EXEC, AGENT_QUERY
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.executor import (
    EXECUTION_ERROR,
    OUTPUT_VALIDATION_ERROR,
    PERMISSION_DENIED,
    SKILL_NOT_FOUND,
    TIMEOUT,
    VALIDATION_ERROR,
    skill_executor,
)
from ecom_agent_matrix.core.skill.skill_registry import (
    SkillExecutionContext,
    exec_skill,
    lookup_skill,
    register_skill,
    skill_execution_context,
)
from ecom_agent_matrix.modules.skills.calc_tool import ProfitCalcTool


@register_skill
class _SlowContractSkill(BaseSkill):
    skill_name = "test_phase2a_slow"
    skill_desc = "timeout contract test"
    read_only = True
    side_effect = False
    risk_level = "low"
    timeout_seconds = 0.01

    async def run(self, params: dict) -> SkillResult:
        await asyncio.sleep(1)
        return SkillResult(success=True)


@register_skill
class _ExplodingContractSkill(BaseSkill):
    skill_name = "test_phase2a_exploding"
    skill_desc = "exception normalization test"
    read_only = True
    side_effect = False
    risk_level = "low"

    async def run(self, params: dict) -> SkillResult:
        raise RuntimeError("internal secret must not leak")


class _ExpectedOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


@register_skill
class _BadOutputContractSkill(BaseSkill):
    skill_name = "test_phase2a_bad_output"
    skill_desc = "output contract test"
    read_only = True
    side_effect = False
    risk_level = "low"
    output_model = _ExpectedOutput

    async def run(self, params: dict) -> SkillResult:
        return SkillResult(success=True, data={"unexpected": "value"})


def test_skill_result_mutable_defaults_are_isolated():
    first = SkillResult(success=True)
    second = SkillResult(success=True)
    first.data["x"] = 1
    first.metadata["trace"] = "a"
    assert second.data == {}
    assert second.metadata == {}


def test_unknown_skill_has_structured_error_and_metadata():
    result = asyncio.run(exec_skill("missing_phase2a_skill", {}))
    assert result.success is False
    assert result.error_code == SKILL_NOT_FOUND
    assert result.metadata["skill_name"] == "missing_phase2a_skill"
    assert result.metadata["latency_ms"] >= 0


def test_invalid_and_extra_input_fail_before_skill_run():
    async def scenario():
        with patch.object(
            ProfitCalcTool,
            "run",
            new=AsyncMock(return_value=SkillResult(success=True)),
        ) as run:
            invalid = await exec_skill(
                "profit_calc",
                {"cost": -1, "shipping": 2, "commission_rate": 0.1, "sell_price": 20},
            )
            extra = await exec_skill(
                "profit_calc",
                {
                    "cost": 10,
                    "shipping": 2,
                    "commission_rate": 0.1,
                    "sell_price": 20,
                    "typo_field": True,
                },
            )
        assert invalid.error_code == VALIDATION_ERROR
        assert extra.error_code == VALIDATION_ERROR
        run.assert_not_awaited()

    asyncio.run(scenario())


def test_permission_matrix_and_explicit_executor_context():
    async def scenario():
        params = {"target_sku": "SKU-1", "competitor": "Temu", "compete_price": 80}
        no_context = await exec_skill("record_competitor_price", params)
        with skill_execution_context(AGENT_QUERY):
            query_result = await exec_skill("record_competitor_price", params)

        with patch(
            "ecom_agent_matrix.modules.skills.price_monitor.AsyncPGClient.execute_sql",
            new=AsyncMock(return_value=[[9]]),
        ):
            exec_result = await skill_executor.execute(
                "record_competitor_price",
                params,
                context=SkillExecutionContext(agent_id=AGENT_EXEC),
            )

        assert no_context.error_code == PERMISSION_DENIED
        assert query_result.error_code == PERMISSION_DENIED
        assert exec_result.success is True
        assert exec_result.data["record_id"] == 9
        assert exec_result.metadata["agent_id"] == AGENT_EXEC

    asyncio.run(scenario())


def test_timeout_is_standardized_without_retry():
    result = asyncio.run(exec_skill("test_phase2a_slow", {}))
    assert result.success is False
    assert result.error_code == TIMEOUT
    assert result.metadata["skill_name"] == "test_phase2a_slow"


def test_execution_exception_is_standardized_and_not_leaked():
    result = asyncio.run(exec_skill("test_phase2a_exploding", {}))
    assert result.success is False
    assert result.error_code == EXECUTION_ERROR
    assert "internal secret" not in result.error_msg


def test_output_contract_failure_is_standardized():
    result = asyncio.run(exec_skill("test_phase2a_bad_output", {}))
    assert result.success is False
    assert result.error_code == OUTPUT_VALIDATION_ERROR


def test_profit_calc_contract_keeps_legal_call_compatible():
    result = asyncio.run(
        exec_skill(
            "profit_calc",
            {"cost": 10, "shipping": 2, "commission_rate": 0.1, "sell_price": 20},
        )
    )
    assert result.success is True
    assert result.data["gross_profit"] == 6
    assert result.metadata["skill_name"] == "profit_calc"


def test_goods_sku_search_contract_keeps_legacy_query_alias():
    candidate = {
        "sku": "SKU-BAG-001",
        "title_zh": "防水背包",
        "title_en": "Waterproof Backpack",
        "category": "bags",
        "price": 29.9,
        "stock_num": 20,
        "match_mode": "literal_trgm",
    }

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.skills.goods_sku_search._literal_trgm_search",
            new=AsyncMock(return_value=[candidate]),
        ):
            return await exec_skill("goods_sku_search", {"query": "防水背包", "top_k": 3})

    result = asyncio.run(scenario())
    assert result.success is True
    assert result.data["best_sku"] == "SKU-BAG-001"
    assert result.data["product_name"] == "防水背包"


def test_four_representative_skills_expose_contracts():
    for name in (
        "profit_calc",
        "goods_sku_search",
        "price_monitor",
        "record_competitor_price",
    ):
        spec = lookup_skill(name).spec()
        assert spec.input_model is not None
        assert spec.output_model is not None
        assert spec.timeout_seconds > 0


def test_register_skill_rejects_invalid_metadata_and_duplicates():
    class ContradictorySkill(BaseSkill):
        skill_name = "test_phase2a_contradictory"
        skill_desc = "invalid"
        read_only = True
        side_effect = True
        risk_level = "low"

        async def run(self, params: dict) -> SkillResult:
            return SkillResult(success=True)

    try:
        register_skill(ContradictorySkill)
        raise AssertionError("contradictory metadata should fail")
    except ValueError:
        pass

    try:
        register_skill(ProfitCalcTool)
        raise AssertionError("duplicate skill_name should fail")
    except ValueError as exc:
        assert "重复注册" in str(exc)

    class InvalidRiskSkill(BaseSkill):
        skill_name = "test_phase2a_invalid_risk"
        skill_desc = "invalid"
        read_only = True
        side_effect = False
        risk_level = "extreme"

        async def run(self, params: dict) -> SkillResult:
            return SkillResult(success=True)

    try:
        register_skill(InvalidRiskSkill)
        raise AssertionError("invalid risk_level should fail")
    except ValueError:
        pass

    class BlankNameSkill(BaseSkill):
        skill_name = ""
        skill_desc = "invalid"
        read_only = True
        side_effect = False
        risk_level = "low"

        async def run(self, params: dict) -> SkillResult:
            return SkillResult(success=True)

    try:
        register_skill(BlankNameSkill)
        raise AssertionError("blank skill_name should fail")
    except ValueError:
        pass
