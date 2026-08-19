"""第一轮 P0 企业级重构回归测试。"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.config.constants import AGENT_EXEC, AGENT_MASTER, AGENT_QUERY, AGENT_RAG
from ecom_agent_matrix.core.llm import ChatResult
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.reply import build_reply
from ecom_agent_matrix.core.mcp.task_waiter import TaskReplyWaiter
from ecom_agent_matrix.core.skill.skill_registry import exec_skill, skill_execution_context
from ecom_agent_matrix.modules.agent_cluster.master_agent import _react_call_one, process_master_task
from ecom_agent_matrix.modules.agent_cluster.master_planner import (
    PlanResult,
    ReactDecision,
    plan_sub_tasks_keyword,
    plan_sub_tasks_llm,
)


def _request(root_id: str, correlation_id: str) -> MCPMessage:
    return MCPMessage(
        task_id=root_id,
        correlation_id=correlation_id,
        sender=AGENT_MASTER,
        target=AGENT_QUERY,
        content={"query": "test"},
    )


def test_same_root_task_different_correlations_do_not_cross_replies():
    async def scenario():
        root_id = "root-shared"
        TaskReplyWaiter.begin("corr-a", 1)
        TaskReplyWaiter.begin("corr-b", 1)

        reply_b = build_reply(_request(root_id, "corr-b"), AGENT_QUERY, success=True)
        reply_a = build_reply(_request(root_id, "corr-a"), AGENT_QUERY, success=True)
        assert TaskReplyWaiter.submit_reply(reply_b) is True
        assert TaskReplyWaiter.submit_reply(reply_a) is True

        got_a, got_b = await asyncio.gather(
            TaskReplyWaiter.wait("corr-a", 0.1),
            TaskReplyWaiter.wait("corr-b", 0.1),
        )
        assert [m.correlation_id for m in got_a] == ["corr-a"]
        assert [m.correlation_id for m in got_b] == ["corr-b"]
        assert all(m.task_id == root_id for m in got_a + got_b)

    asyncio.run(scenario())


def test_late_timed_out_reply_is_not_consumed_by_next_react_step():
    async def scenario():
        root_id = "root-timeout"
        TaskReplyWaiter.begin("corr-old", 1)
        assert await TaskReplyWaiter.wait("corr-old", 0.001) == []

        TaskReplyWaiter.begin("corr-new", 1)
        late = build_reply(_request(root_id, "corr-old"), AGENT_QUERY, success=True)
        current = build_reply(_request(root_id, "corr-new"), AGENT_QUERY, success=True)
        assert TaskReplyWaiter.submit_reply(late) is False
        assert TaskReplyWaiter.submit_reply(current) is True
        replies = await TaskReplyWaiter.wait("corr-new", 0.1)
        assert [m.correlation_id for m in replies] == ["corr-new"]

    asyncio.run(scenario())


def test_each_react_call_agent_creates_new_correlation_id():
    async def scenario():
        seen: list[str] = []

        async def fake_dispatch(task_id, correlation_id, target_agent, payload, priority):
            seen.append(correlation_id)
            reply = build_reply(
                _request(task_id, correlation_id),
                target_agent,
                success=True,
                data={"answer": "ok"},
            )
            TaskReplyWaiter.submit_reply(reply)

        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master_agent._dispatch_subtask",
            new=AsyncMock(side_effect=fake_dispatch),
        ):
            await _react_call_one("root", AGENT_QUERY, {}, 1)
            await _react_call_one("root", AGENT_QUERY, {}, 1)
        assert len(seen) == 2
        assert seen[0] != seen[1]

    asyncio.run(scenario())


def test_unknown_request_returns_clarify_without_rag_dispatch():
    plan = plan_sub_tasks_keyword({"query": "随便说点什么 xyz"}, [])
    assert plan.decision == "clarify"
    assert plan.sub_tasks == []
    assert plan.planner == "clarify"

    async def scenario():
        request = MCPMessage(
            task_id="root-clarify",
            sender="api_gateway",
            target=AGENT_MASTER,
            content={"query": "随便说点什么 xyz"},
        )
        long_mem = AsyncMock()
        long_mem.recall.return_value = []
        clarify = PlanResult(
            sub_tasks=[],
            plan_confidence=0.3,
            reasoning="无法可靠识别",
            planner="clarify",
            decision="clarify",
            clarification_question="请补充具体需求。",
        )
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master_agent.plan_sub_tasks_llm",
            new=AsyncMock(return_value=clarify),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master_agent._dispatch_subtask",
            new=AsyncMock(),
        ) as dispatch, patch(
            "ecom_agent_matrix.modules.agent_cluster.master_agent.mcp_bus.send_msg",
            new=AsyncMock(return_value=True),
        ) as send:
            await process_master_task(request, long_mem)
        dispatch.assert_not_awaited()
        sent = send.await_args.args[0]
        assert sent.content["data"]["mode"] == "clarify"
        assert sent.content["data"]["summary"] == "请补充具体需求。"

    asyncio.run(scenario())


def test_unknown_request_does_not_call_llm_or_rag_when_llm_is_configured():
    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master_planner.is_llm_configured",
            return_value=True,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master_planner.llm_chat",
            new=AsyncMock(),
        ) as llm:
            plan = await plan_sub_tasks_llm({"query": "??? xyz"}, [])
        llm.assert_not_awaited()
        assert plan.decision == "clarify"
        assert plan.sub_tasks == []

    asyncio.run(scenario())


def test_refund_rules_route_rag_and_refund_reply_routes_exec_crm():
    rules = plan_sub_tasks_keyword({"query": "退款规则是什么"}, [])
    reply = plan_sub_tasks_keyword({"query": "帮我回复退款客户"}, [])
    order = plan_sub_tasks_keyword({"query": "查询订单数据"}, [])
    assert [x["target_agent"] for x in rules.sub_tasks] == [AGENT_RAG]
    assert [x["target_agent"] for x in reply.sub_tasks] == [AGENT_EXEC]
    assert [x["target_agent"] for x in order.sub_tasks] == [AGENT_QUERY]
    assert reply.sub_tasks[0]["payload"]["_inferred_task_type"] == "customer_service"


def test_query_context_rejects_write_skill():
    async def scenario():
        with skill_execution_context(AGENT_QUERY):
            result = await exec_skill(
                "order_risk_check",
                {"order_no": "ORD-1", "total_amount": 999, "buy_count": 30},
            )
        assert result.success is False
        assert "无权执行非只读" in result.error_msg

    asyncio.run(scenario())


def test_plan_reason_never_contains_internal_reasoning_content():
    async def scenario():
        model_result = ChatResult(
            content=json.dumps(
                {
                    "decision": "dispatch",
                    "agents": [AGENT_EXEC],
                    "confidence": 0.95,
                    "reasoning": "用户明确要求优化广告",
                }
            ),
            reasoning_content="TOP SECRET INTERNAL CHAIN",
        )
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master_planner.is_llm_configured",
            return_value=True,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master_planner.llm_chat",
            new=AsyncMock(return_value=model_result),
        ):
            plan = await plan_sub_tasks_llm({"query": "优化广告投放"}, [])
        assert plan.reasoning == "用户明确要求优化广告"
        assert "TOP SECRET" not in plan.reasoning

    asyncio.run(scenario())


def test_master_react_trace_does_not_persist_reasoning_content():
    async def scenario():
        request = MCPMessage(
            task_id="root-reasoning",
            sender="api_gateway",
            target=AGENT_MASTER,
            content={"query": "查订单"},
        )
        plan = PlanResult(
            sub_tasks=[{"target_agent": AGENT_QUERY, "payload": {"query": "查订单"}}],
            plan_confidence=0.95,
            reasoning="订单查询",
            planner="test",
        )
        decisions = [
            ReactDecision(
                thought="查询订单",
                action="call_agent",
                agent=AGENT_QUERY,
                payload={"query": "查订单"},
                reasoning_content="NEVER STORE THIS",
            ),
            ReactDecision(
                thought="完成",
                action="finish",
                final_answer="完成",
                reasoning_content="NEVER STORE THIS EITHER",
            ),
        ]
        long_mem = AsyncMock()
        long_mem.recall.return_value = []
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.master_agent.plan_sub_tasks_llm",
            new=AsyncMock(return_value=plan),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master_agent.react_decide",
            new=AsyncMock(side_effect=decisions),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master_agent._react_call_one",
            new=AsyncMock(
                return_value={
                    "agent": AGENT_QUERY,
                    "success": True,
                    "data": {"answer": "完成"},
                    "error_msg": "",
                    "timed_out": False,
                }
            ),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master_agent.polish_final_output",
            new=AsyncMock(return_value="完成"),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.master_agent.mcp_bus.send_msg",
            new=AsyncMock(return_value=True),
        ) as send:
            await process_master_task(request, long_mem)

        persisted_reply = send.await_args_list[0].args[0].model_dump()
        assert "reasoning_content" not in json.dumps(persisted_reply, ensure_ascii=False)
        if long_mem.safe_save_memory.await_count:
            saved_content = long_mem.safe_save_memory.await_args.kwargs["content"]
            assert "reasoning_content" not in saved_content
            assert "NEVER STORE" not in saved_content

    asyncio.run(scenario())
