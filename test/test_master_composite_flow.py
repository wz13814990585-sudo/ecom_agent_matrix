from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.config.constants import AGENT_EXEC, AGENT_MASTER, AGENT_QUERY, AGENT_RAG
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.reply import build_reply
from ecom_agent_matrix.core.mcp.task_waiter import TaskReplyWaiter
from ecom_agent_matrix.core.skill.base_skill import SkillResult
from ecom_agent_matrix.modules.agent_cluster.handlers.crm import run_crm_workflow
from ecom_agent_matrix.modules.agent_cluster.master.planner import build_composite_plan
from ecom_agent_matrix.modules.agent_cluster.master.telemetry import MasterLLMTelemetry
from ecom_agent_matrix.modules.agent_cluster.master_agent import process_master_task

QUERY = "根据 ORD-123 的订单状态和退款规则帮我回复客户"


def test_composite_template_has_parallel_context_and_dependent_crm():
    plan = build_composite_plan({"query": QUERY})
    assert plan is not None
    assert plan.planner_source == "rules_composite"
    by_id = {step.step_id: step for step in plan.steps}
    assert by_id["order_context"].agent == AGENT_QUERY
    assert by_id["policy_context"].agent == AGENT_RAG
    assert by_id["customer_reply"].agent == AGENT_EXEC
    assert by_id["customer_reply"].depends_on == ["order_context", "policy_context"]


def test_composite_master_flow_passes_context_and_uses_final_answer():
    async def scenario():
        request = MCPMessage(
            task_id="root-composite",
            correlation_id="gateway-correlation",
            sender="api_gateway",
            target=AGENT_MASTER,
            content={"query": QUERY},
        )
        sent_children: list[MCPMessage] = []
        final_replies: list[MCPMessage] = []

        async def send(message: MCPMessage):
            if message.sender == AGENT_MASTER and message.target in {
                AGENT_QUERY,
                AGENT_RAG,
                AGENT_EXEC,
            }:
                sent_children.append(message)
                if message.target == AGENT_EXEC:
                    assert set(message.content["_upstream_context"]) == {
                        "order_context",
                        "policy_context",
                    }
                    data = {"answer": "根据订单状态和退款规则生成的客服回复"}
                elif message.target == AGENT_QUERY:
                    data = {"order_no": "ORD-123", "status": "shipped"}
                else:
                    data = {"answer": "符合条件可退款", "docs": [{"title": "refund"}]}
                TaskReplyWaiter.submit_reply(
                    build_reply(message, sender=message.target, success=True, data=data)
                )
            else:
                final_replies.append(message)
            return True

        memory = AsyncMock()
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master.executor.mcp_bus.send_msg",
            new=AsyncMock(side_effect=send),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master_agent.mcp_bus.send_msg",
            new=AsyncMock(side_effect=send),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master_agent.polish_final_output",
            new=AsyncMock(),
        ) as polish, patch(
            "ecom_agent_matrix.modules.agent_cluster.master_agent.recovery_controller.run",
            new=AsyncMock(),
        ) as recovery:
            await process_master_task(request, memory)
        return sent_children, final_replies[-1], polish, recovery

    children, reply, polish, recovery = asyncio.run(scenario())
    data = reply.content["data"]
    assert data["mode"] == "plan"
    assert data["summary"] == "根据订单状态和退款规则生成的客服回复"
    assert data["master_llm_usage"]["calls"] == 0
    assert [message.target for message in children][-1] == AGENT_EXEC
    polish.assert_not_awaited()
    recovery.assert_not_awaited()


def test_crm_upstream_policy_disables_duplicate_rag():
    async def scenario():
        calls: list[tuple[str, dict]] = []

        async def exec_skill(name: str, params: dict):
            calls.append((name, params))
            if name == "crm_reply":
                return SkillResult(
                    success=True,
                    data={
                        "answer": "verified answer",
                        "llm_ok": True,
                        "rag_used": False,
                        "rag_doc_count": 0,
                        "rag_error": "",
                    },
                )
            return SkillResult(success=True, data={})

        task = {
            "task_id": "root-crm",
            "query": QUERY,
            "task_type": "customer_service",
            "_upstream_context": {
                "order_context": {
                    "task_type": "order_query",
                    "data": {"status": "shipped"},
                },
                "policy_context": {
                    "task_type": "knowledge_qa",
                    "data": {"answer": "refund allowed"},
                },
            },
        }
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.exec_skill",
            new=AsyncMock(side_effect=exec_skill),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.AgentShortMemory",
            side_effect=RuntimeError("memory unavailable"),
        ):
            result = await run_crm_workflow(task)
        return result, calls

    result, calls = asyncio.run(scenario())
    assert result.success is True
    crm_params = next(params for name, params in calls if name == "crm_reply")
    assert crm_params["use_rag"] is False
    assert crm_params["upstream_context"]["order_context"]["data"]["status"] == "shipped"
    assert crm_params["upstream_context"]["policy_context"]["data"]["answer"] == "refund allowed"
