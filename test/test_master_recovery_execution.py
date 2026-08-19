from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ecom_agent_matrix.config.constants import AGENT_EXEC, AGENT_MASTER, AGENT_QUERY, AGENT_RAG
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.reply import build_reply
from ecom_agent_matrix.core.mcp.task_waiter import TaskReplyWaiter
from ecom_agent_matrix.modules.agent_cluster.master.executor import MasterPlanExecutor
from ecom_agent_matrix.modules.agent_cluster.master.recovery import (
    RecoveryApplication,
    apply_recovery_decision,
    build_replan_input,
)
from ecom_agent_matrix.modules.agent_cluster.master.schemas import (
    MasterPlan,
    PlanExecutionResult,
    PlanStep,
    RecoveryDecision,
    StepResult,
)
from ecom_agent_matrix.modules.agent_cluster.master.telemetry import MasterLLMTelemetry
from ecom_agent_matrix.modules.agent_cluster.master_router import MasterRouteDecision
from ecom_agent_matrix.modules.agent_cluster import master_agent as master_module


def _root() -> MCPMessage:
    return MCPMessage(
        task_id="root-recovery",
        correlation_id="gateway-correlation",
        sender="api_gateway",
        target=AGENT_MASTER,
        content={"query": "根据 ORD-123 的订单状态和退款规则帮我回复客户"},
    )


def _plan() -> MasterPlan:
    return MasterPlan(
        decision="execute",
        confidence=0.95,
        reason_code="RECOVERY_TEST",
        planner_source="test",
        steps=[
            PlanStep(step_id="order_context", agent=AGENT_QUERY, task_type="order_query"),
            PlanStep(step_id="policy_context", agent=AGENT_RAG, task_type="knowledge_qa"),
            PlanStep(
                step_id="customer_reply",
                agent=AGENT_EXEC,
                task_type="customer_service",
                depends_on=["order_context", "policy_context"],
            ),
        ],
    )


def _failed_execution(step_id: str, agent: str, task_type: str) -> PlanExecutionResult:
    return PlanExecutionResult(
        all_success=False,
        partial_success=False,
        timed_out=False,
        step_results={
            step_id: StepResult(
                step_id=step_id,
                agent=agent,
                task_type=task_type,
                status="FAILED",
                error_code="AGENT_FAILED",
            )
        },
    )


@pytest.mark.parametrize(
    ("retry_step", "retry_agent"),
    [("order_context", AGENT_QUERY), ("policy_context", AGENT_RAG)],
)
def test_read_retry_resumes_dag_without_repeating_success(retry_step, retry_agent):
    async def scenario():
        counts = {AGENT_QUERY: 0, AGENT_RAG: 0, AGENT_EXEC: 0}
        messages: list[MCPMessage] = []

        async def send(message: MCPMessage):
            messages.append(message)
            counts[message.target] += 1
            if message.target == retry_agent and counts[message.target] == 1:
                return True
            data = {"answer": "final"} if message.target == AGENT_EXEC else {"context": True}
            TaskReplyWaiter.submit_reply(
                build_reply(message, sender=message.target, success=True, data=data)
            )
            return True

        executor = MasterPlanExecutor(max_concurrent=2, timeout=0.01)
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master.executor.mcp_bus.send_msg",
            new=AsyncMock(side_effect=send),
        ):
            first = await executor.execute(_plan(), _root())
            applied = await apply_recovery_decision(
                RecoveryDecision(
                    action="retry_agent",
                    step_id=retry_step,
                    reason_code="RETRY_TIMEOUT",
                ),
                plan=_plan(),
                execution=first,
                root_message=_root(),
                task_input=_root().content,
                executor=executor,
                planner=AsyncMock(),
                telemetry=MasterLLMTelemetry(),
            )
        return first, applied, counts, messages

    first, applied, counts, messages = asyncio.run(scenario())
    assert first.step_results[retry_step].status == "FAILED"
    assert first.step_results["customer_reply"].status == "SKIPPED"
    assert applied.execution.all_success is True
    assert counts[retry_agent] == 2
    other = AGENT_RAG if retry_agent == AGENT_QUERY else AGENT_QUERY
    assert counts[other] == 1
    assert counts[AGENT_EXEC] == 1
    retried = [message for message in messages if message.target == retry_agent]
    assert retried[0].correlation_id != retried[1].correlation_id
    assert {message.task_id for message in messages} == {"root-recovery"}


def test_biz_exec_retry_is_rejected_without_dispatch():
    async def scenario():
        executor = AsyncMock()
        execution = _failed_execution("customer_reply", AGENT_EXEC, "customer_service")
        applied = await apply_recovery_decision(
            RecoveryDecision(
                action="retry_agent",
                step_id="customer_reply",
                reason_code="RETRY",
            ),
            plan=_plan(),
            execution=execution,
            root_message=_root(),
            task_input=_root().content,
            executor=executor,
            planner=AsyncMock(),
            telemetry=MasterLLMTelemetry(),
        )
        return applied, executor

    applied, executor = asyncio.run(scenario())
    assert applied.decision.action == "finish"
    assert applied.decision.reason_code == "UNSAFE_RECOVERY_RETRY_REJECTED"
    executor.resume.assert_not_awaited()


def test_replan_is_rejected_after_exec_may_have_run():
    async def scenario():
        planner = AsyncMock()
        executor = AsyncMock()
        applied = await apply_recovery_decision(
            RecoveryDecision(action="replan", reason_code="REPLAN"),
            plan=_plan(),
            execution=_failed_execution("customer_reply", AGENT_EXEC, "customer_service"),
            root_message=_root(),
            task_input=_root().content,
            executor=executor,
            planner=planner,
            telemetry=MasterLLMTelemetry(),
        )
        return applied, planner, executor

    applied, planner, executor = asyncio.run(scenario())
    assert applied.decision.action == "finish"
    assert applied.decision.reason_code == "UNSAFE_REPLAN_AFTER_EXEC"
    planner.plan.assert_not_awaited()
    executor.resume.assert_not_awaited()


def test_safe_replan_uses_compact_context_revalidates_and_executes():
    async def scenario():
        first = PlanExecutionResult(
            all_success=False,
            partial_success=True,
            timed_out=True,
            step_results={
                "order_context": StepResult(
                    step_id="order_context",
                    agent=AGENT_QUERY,
                    task_type="order_query",
                    status="FAILED",
                    error_code="TIMEOUT",
                    data={"rows": ["MUST_NOT_REPLAN"]},
                ),
                "policy_context": StepResult(
                    step_id="policy_context",
                    agent=AGENT_RAG,
                    task_type="knowledge_qa",
                    status="SUCCESS",
                    success=True,
                    data={"docs": ["FULL_RAG_DOC"]},
                ),
                "customer_reply": StepResult(
                    step_id="customer_reply",
                    agent=AGENT_EXEC,
                    task_type="customer_service",
                    status="SKIPPED",
                    error_code="DEPENDENCY_FAILED",
                ),
            },
        )
        planner = AsyncMock()
        planner.plan.return_value = _plan()
        final = PlanExecutionResult(
            all_success=True,
            partial_success=False,
            timed_out=False,
            step_results={
                step.step_id: StepResult(
                    step_id=step.step_id,
                    agent=step.agent,
                    task_type=step.task_type,
                    status="SUCCESS",
                    success=True,
                )
                for step in _plan().steps
            },
        )
        executor = AsyncMock()
        executor.resume.return_value = final
        task_input = {
            "query": "complex token=TOP_SECRET",
            "history": ["MUST_NOT_REPLAN"],
            "api_key": "MUST_NOT_REPLAN",
        }
        applied = await apply_recovery_decision(
            RecoveryDecision(action="replan", reason_code="REPLAN"),
            plan=_plan(),
            execution=first,
            root_message=_root(),
            task_input=task_input,
            executor=executor,
            planner=planner,
            telemetry=MasterLLMTelemetry(),
        )
        return applied, planner, executor

    applied, planner, executor = asyncio.run(scenario())
    assert applied.execution.all_success is True
    replanning_input = planner.plan.await_args.args[0]
    serialized = str(replanning_input)
    assert "TOP_SECRET" not in serialized
    assert "MUST_NOT_REPLAN" not in serialized
    assert "FULL_RAG_DOC" not in serialized
    assert replanning_input["_recovery_context"]["successful_step_ids"] == ["policy_context"]
    assert executor.resume.await_count == 1
    assert executor.resume.await_args.kwargs["retry_step_ids"] == {
        "order_context",
        "customer_reply",
    }


def test_invalid_replanned_plan_is_not_executed():
    async def scenario():
        first = _failed_execution("order_context", AGENT_QUERY, "order_query")
        invalid = _plan().model_copy(
            update={
                "steps": [
                    _plan().steps[0].model_copy(update={"depends_on": ["missing_step"]})
                ]
            }
        )
        planner = AsyncMock()
        planner.plan.return_value = invalid
        executor = AsyncMock()
        applied = await apply_recovery_decision(
            RecoveryDecision(action="replan", reason_code="REPLAN"),
            plan=_plan(),
            execution=first,
            root_message=_root(),
            task_input=_root().content,
            executor=executor,
            planner=planner,
            telemetry=MasterLLMTelemetry(),
        )
        return applied, executor

    applied, executor = asyncio.run(scenario())
    assert applied.decision.action == "finish"
    assert applied.decision.reason_code == "INVALID_REPLANNED_PLAN"
    executor.resume.assert_not_awaited()


@pytest.mark.parametrize("action", ["finish", "clarify"])
def test_terminal_recovery_actions_do_not_execute_agents(action):
    async def scenario():
        executor = AsyncMock()
        applied = await apply_recovery_decision(
            RecoveryDecision(
                action=action,
                reason_code="TERMINAL",
                clarification_question="please clarify" if action == "clarify" else "",
            ),
            plan=_plan(),
            execution=_failed_execution("order_context", AGENT_QUERY, "order_query"),
            root_message=_root(),
            task_input=_root().content,
            executor=executor,
            planner=AsyncMock(),
            telemetry=MasterLLMTelemetry(),
        )
        return applied, executor

    applied, executor = asyncio.run(scenario())
    assert applied.decision.action == action
    executor.resume.assert_not_awaited()


def test_master_recovery_loop_is_bounded_and_final_status_is_authoritative(monkeypatch):
    async def scenario():
        request = _root()
        first = _failed_execution("order_context", AGENT_QUERY, "order_query")
        still_failed = RecoveryApplication(
            plan=_plan(),
            execution=first,
            decision=RecoveryDecision(
                action="retry_agent",
                step_id="order_context",
                reason_code="RETRY",
            ),
            continue_recovery=True,
            execution_changed=True,
        )
        route = MasterRouteDecision(
            mode="planner", confidence=0.5, reason_code="TEST"
        )
        memory = AsyncMock()
        monkeypatch.setattr(settings, "MASTER_RECOVERY_MAX_STEPS", 2)
        with patch.object(master_module, "route_master_task", return_value=route), patch.object(
            master_module.typed_master_planner, "plan", new=AsyncMock(return_value=_plan())
        ), patch.object(
            master_module.MasterPlanExecutor, "execute", new=AsyncMock(return_value=first)
        ), patch.object(
            master_module.recovery_controller,
            "run",
            new=AsyncMock(
                return_value=RecoveryDecision(
                    action="retry_agent",
                    step_id="order_context",
                    reason_code="RETRY",
                )
            ),
        ) as controller, patch.object(
            master_module,
            "apply_recovery_decision",
            new=AsyncMock(return_value=still_failed),
        ) as apply, patch.object(
            master_module.mcp_bus, "send_msg", new=AsyncMock(return_value=True)
        ) as send:
            await master_module.process_master_task(request, memory)
        return controller, apply, send.await_args_list[0].args[0].content["data"]

    controller, apply, data = asyncio.run(scenario())
    assert controller.await_count == 2
    assert apply.await_count == 2
    assert len(data["recovery"]["actions"]) == 2
    assert data["all_success"] is False
    assert data["recovery"]["final_status"] == "FAILED"


def test_master_returns_recovered_execution_as_final_status():
    async def scenario():
        request = _root()
        child_counts = {AGENT_QUERY: 0, AGENT_RAG: 0, AGENT_EXEC: 0}
        final_replies: list[MCPMessage] = []

        async def send(message: MCPMessage):
            if message.target in child_counts:
                child_counts[message.target] += 1
                if message.target == AGENT_QUERY and child_counts[AGENT_QUERY] == 1:
                    return True
                TaskReplyWaiter.submit_reply(
                    build_reply(
                        message,
                        sender=message.target,
                        success=True,
                        data={"answer": "recovered final"}
                        if message.target == AGENT_EXEC
                        else {"context": True},
                    )
                )
            else:
                final_replies.append(message)
            return True

        route = MasterRouteDecision(
            mode="planner", confidence=0.9, reason_code="TEST"
        )
        memory = AsyncMock()
        with patch.object(master_module, "route_master_task", return_value=route), patch.object(
            master_module.typed_master_planner, "plan", new=AsyncMock(return_value=_plan())
        ), patch.object(
            master_module.recovery_controller,
            "run",
            new=AsyncMock(
                return_value=RecoveryDecision(
                    action="retry_agent",
                    step_id="order_context",
                    reason_code="RETRY_TIMEOUT",
                )
            ),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master.executor.settings.MCP_TIMEOUT",
            0.01,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master.executor.mcp_bus.send_msg",
            new=AsyncMock(side_effect=send),
        ), patch.object(
            master_module.mcp_bus, "send_msg", new=AsyncMock(side_effect=send)
        ):
            await master_module.process_master_task(request, memory)
        return final_replies[-1].content["data"], child_counts

    data, counts = asyncio.run(scenario())
    assert counts == {AGENT_QUERY: 2, AGENT_RAG: 1, AGENT_EXEC: 1}
    assert data["all_success"] is True
    assert data["partial_success"] is False
    assert data["timed_out"] is False
    assert data["summary"] == "recovered final"
    assert data["recovery"]["final_status"] == "SUCCESS"


def test_replan_input_contains_only_allowed_compact_fields():
    execution = _failed_execution("order_context", AGENT_QUERY, "order_query")
    value = build_replan_input(
        {"query": "query", "password": "secret", "history": ["raw"]}, execution
    )
    assert set(value) == {"query", "_recovery_context"}
    assert set(value["_recovery_context"]) == {"failed_steps", "successful_step_ids"}
