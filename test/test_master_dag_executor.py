from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.config.constants import AGENT_EXEC, AGENT_MASTER, AGENT_QUERY, AGENT_RAG
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.reply import build_reply
from ecom_agent_matrix.core.mcp.task_waiter import TaskReplyWaiter
from ecom_agent_matrix.modules.agent_cluster.master.executor import MasterPlanExecutor
from ecom_agent_matrix.modules.agent_cluster.master.schemas import MasterPlan, PlanStep


def _root() -> MCPMessage:
    return MCPMessage(
        task_id="root-dag",
        correlation_id="gateway-correlation",
        sender="api_gateway",
        target=AGENT_MASTER,
        content={"query": "composite"},
    )


def _plan() -> MasterPlan:
    return MasterPlan(
        decision="execute",
        confidence=0.99,
        reason_code="TEST_DAG",
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


def test_independent_steps_are_concurrent_and_exec_waits_for_both():
    async def scenario():
        active = 0
        max_active = 0
        completed: set[str] = set()
        sent: list[MCPMessage] = []

        async def send(message: MCPMessage):
            nonlocal active, max_active
            sent.append(message)
            if message.target == AGENT_EXEC:
                assert completed == {AGENT_QUERY, AGENT_RAG}
                assert set(message.content["_upstream_context"]) == {
                    "order_context",
                    "policy_context",
                }
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            completed.add(message.target)
            active -= 1
            TaskReplyWaiter.submit_reply(
                build_reply(
                    message,
                    sender=message.target,
                    success=True,
                    data={"answer": "final" if message.target == AGENT_EXEC else "context"},
                )
            )
            return True

        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master.executor.mcp_bus.send_msg",
            new=AsyncMock(side_effect=send),
        ):
            result = await MasterPlanExecutor(max_concurrent=2, timeout=0.5).execute(
                _plan(), _root()
            )
        return result, sent, max_active

    result, sent, max_active = asyncio.run(scenario())
    assert result.all_success is True
    assert max_active == 2
    assert sent[-1].target == AGENT_EXEC
    assert len({message.correlation_id for message in sent}) == 3
    assert {message.task_id for message in sent} == {"root-dag"}


def test_concurrency_limit_covers_dispatch_and_wait():
    async def scenario():
        active = 0
        max_active = 0

        async def send(message: MCPMessage):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            TaskReplyWaiter.submit_reply(
                build_reply(message, sender=message.target, success=True, data={})
            )
            active -= 1
            return True

        plan = _plan().model_copy(update={"steps": _plan().steps[:2]})
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master.executor.mcp_bus.send_msg",
            new=AsyncMock(side_effect=send),
        ):
            await MasterPlanExecutor(max_concurrent=1, timeout=0.5).execute(plan, _root())
        return max_active

    assert asyncio.run(scenario()) == 1


def test_required_dependency_failure_skips_downstream():
    async def scenario():
        sent: list[MCPMessage] = []

        async def send(message: MCPMessage):
            sent.append(message)
            success = message.target != AGENT_QUERY
            TaskReplyWaiter.submit_reply(
                build_reply(message, sender=message.target, success=success, data={})
            )
            return True

        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master.executor.mcp_bus.send_msg",
            new=AsyncMock(side_effect=send),
        ):
            result = await MasterPlanExecutor(timeout=0.2).execute(_plan(), _root())
        return result, sent

    result, sent = asyncio.run(scenario())
    final = result.step_results["customer_reply"]
    assert final.status == "SKIPPED"
    assert final.error_code == "DEPENDENCY_FAILED"
    assert AGENT_EXEC not in {message.target for message in sent}
    assert result.partial_success is True


def test_timeout_is_propagated_and_waiter_is_cleaned():
    async def scenario():
        plan = _plan().model_copy(update={"steps": [_plan().steps[0]]})
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master.executor.mcp_bus.send_msg",
            new=AsyncMock(return_value=True),
        ):
            result = await MasterPlanExecutor(timeout=0.01).execute(plan, _root())
        return result

    result = asyncio.run(scenario())
    step = result.step_results["order_context"]
    assert step.status == "FAILED"
    assert step.error_code == "TIMEOUT"
    assert TaskReplyWaiter.pending_count(step.correlation_id) == 0
