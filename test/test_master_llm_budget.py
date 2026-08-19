from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.core.llm import ChatResult
from ecom_agent_matrix.modules.agent_cluster.master.planner import TypedMasterPlanner
from ecom_agent_matrix.modules.agent_cluster.master.react import RecoveryController
from ecom_agent_matrix.modules.agent_cluster.master.schemas import PlanExecutionResult, StepResult
from ecom_agent_matrix.modules.agent_cluster.master.telemetry import MasterLLMTelemetry


def test_rule_composite_uses_zero_llm_calls():
    async def scenario():
        telemetry = MasterLLMTelemetry()
        plan = await TypedMasterPlanner().plan(
            {"query": "根据 ORD-123 的订单状态和退款规则帮我回复客户"}, telemetry
        )
        return plan, telemetry.snapshot()

    plan, usage = asyncio.run(scenario())
    assert plan.planner_source == "rules_composite"
    assert usage.calls == 0
    assert usage.planner.calls == 0


def test_real_planner_call_records_provider_tokens():
    async def scenario():
        telemetry = MasterLLMTelemetry()
        response = ChatResult(
            content=json.dumps(
                {
                    "decision": "execute",
                    "confidence": 0.8,
                    "reason_code": "LLM_PLAN",
                    "steps": [
                        {
                            "step_id": "order_context",
                            "agent": "data_query",
                            "task_type": "order_query",
                            "depends_on": [],
                        }
                    ],
                }
            ),
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            reasoning_content="must not persist",
        )
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master.planner.is_llm_configured",
            return_value=True,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master.planner.llm_chat",
            new=AsyncMock(return_value=response),
        ):
            plan = await TypedMasterPlanner().plan({"query": "complex unknown"}, telemetry)
        return plan, telemetry.snapshot().model_dump()

    plan, usage = asyncio.run(scenario())
    assert plan.decision == "execute"
    assert usage["planner"] == {
        "calls": 1,
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }
    assert "reasoning_content" not in json.dumps(plan.model_dump())


def test_invalid_llm_plan_is_not_executable():
    async def scenario():
        telemetry = MasterLLMTelemetry()
        invalid = ChatResult(
            content=json.dumps(
                {
                    "decision": "execute",
                    "confidence": 0.8,
                    "reason_code": "BAD",
                    "steps": [
                        {
                            "step_id": "step_0",
                            "agent": "tool_agent",
                            "task_type": "unknown",
                        }
                    ],
                }
            )
        )
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master.planner.is_llm_configured",
            return_value=True,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master.planner.llm_chat",
            new=AsyncMock(return_value=invalid),
        ):
            return await TypedMasterPlanner().plan({"query": "ambiguous"}, telemetry)

    plan = asyncio.run(scenario())
    assert plan.decision == "clarify"
    assert plan.reason_code == "INVALID_LLM_PLAN"
    assert plan.steps == []


def test_budget_prevents_provider_call():
    async def scenario():
        telemetry = MasterLLMTelemetry(max_calls=0)
        provider = AsyncMock()
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master.planner.is_llm_configured",
            return_value=True,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master.planner.llm_chat",
            new=provider,
        ):
            plan = await TypedMasterPlanner().plan({"query": "complex unknown"}, telemetry)
        return plan, provider, telemetry.snapshot()

    plan, provider, usage = asyncio.run(scenario())
    assert plan.reason_code == "LLM_BUDGET_EXCEEDED"
    provider.assert_not_awaited()
    assert usage.calls == 0


def test_recovery_is_bounded_and_never_retries_exec_write_step(monkeypatch):
    async def scenario():
        failed = PlanExecutionResult(
            all_success=False,
            partial_success=False,
            timed_out=False,
            step_results={
                "customer_reply": StepResult(
                    step_id="customer_reply",
                    agent="biz_exec",
                    task_type="customer_service",
                    status="FAILED",
                    error_code="AGENT_FAILED",
                )
            },
        )
        telemetry = MasterLLMTelemetry(max_calls=3)
        response = ChatResult(
            content=json.dumps(
                {
                    "action": "retry_agent",
                    "step_id": "customer_reply",
                    "reason_code": "RETRY",
                }
            )
        )
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master.react.is_llm_configured",
            return_value=True,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master.react.llm_chat",
            new=AsyncMock(return_value=response),
        ) as provider:
            decision = await RecoveryController().run(failed, telemetry)
        return decision, provider, telemetry.snapshot()

    decision, provider, usage = asyncio.run(scenario())
    assert provider.await_count == 1
    assert decision.action == "finish"
    assert decision.reason_code == "UNSAFE_RECOVERY_RETRY_REJECTED"
    assert usage.recovery.calls == 1


def test_recovery_invalid_outputs_stop_at_configured_max(monkeypatch):
    async def scenario():
        failed = PlanExecutionResult(
            all_success=False,
            partial_success=False,
            timed_out=True,
            step_results={
                "order_context": StepResult(
                    step_id="order_context",
                    agent="data_query",
                    task_type="order_query",
                    status="FAILED",
                    error_code="TIMEOUT",
                )
            },
        )
        telemetry = MasterLLMTelemetry(max_calls=3)
        provider = AsyncMock(return_value=ChatResult(content="not-json"))
        monkeypatch.setattr(
            "ecom_agent_matrix.modules.agent_cluster.master.react.settings.MASTER_RECOVERY_MAX_STEPS",
            2,
        )
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master.react.is_llm_configured",
            return_value=True,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master.react.llm_chat",
            new=provider,
        ):
            decision = await RecoveryController().run(failed, telemetry)
        return decision, provider, telemetry.snapshot()

    decision, provider, usage = asyncio.run(scenario())
    assert provider.await_count == 2
    assert decision.reason_code == "RECOVERY_EXHAUSTED"
    assert usage.recovery.calls == 2
