"""Phase 3A：Master deterministic Fast Path 与 ReAct 完成语义。"""
from __future__ import annotations

import asyncio
import inspect
import json
from unittest.mock import AsyncMock, patch

import pytest

from ecom_agent_matrix.config.constants import AGENT_EXEC, AGENT_MASTER, AGENT_QUERY, AGENT_RAG
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.reply import build_reply
from ecom_agent_matrix.core.mcp.task_waiter import TaskReplyWaiter
from ecom_agent_matrix.modules.agent_cluster import master_agent as master_module
from ecom_agent_matrix.modules.agent_cluster.master_agent import (
    execute_fast_path,
    process_master_task,
)
from ecom_agent_matrix.modules.agent_cluster.master_planner import PlanResult, ReactDecision
from ecom_agent_matrix.modules.agent_cluster.master_router import (
    MasterRouteDecision,
    route_master_task,
)


@pytest.mark.parametrize(
    "task_type,agent",
    [
        ("goods_search", AGENT_QUERY),
        ("stock_analysis", AGENT_QUERY),
        ("knowledge_qa", AGENT_RAG),
        ("customer_service", AGENT_EXEC),
        ("risk_control", AGENT_EXEC),
    ],
)
def test_explicit_task_type_uses_fast_path(task_type, agent):
    route = route_master_task({"task_type": task_type, "query": "ambiguous words ignored"})
    assert route.mode == "fast_path"
    assert route.task_type == task_type
    assert route.target_agents == [agent]
    assert route.confidence == 1
    assert route.reason_code == "EXPLICIT_TASK_TYPE"


@pytest.mark.parametrize(
    "query,task_type,agent,reason_code",
    [
        ("退款规则是什么", "knowledge_qa", AGENT_RAG, "RULE_KNOWLEDGE"),
        ("ORD-123 订单状态", "order_query", AGENT_QUERY, "RULE_ORDER_QUERY"),
        ("查询 SKU-1 库存", "stock_analysis", AGENT_QUERY, "RULE_STOCK"),
        ("优化 Meta 广告", "ad_optimize", AGENT_EXEC, "RULE_AD_OPTIMIZE"),
        ("生成运营日报", "ops_report", AGENT_EXEC, "RULE_REPORT"),
        ("帮我回复退款客户", "customer_service", AGENT_EXEC, "RULE_CRM"),
    ],
)
def test_high_confidence_rule_uses_fast_path(query, task_type, agent, reason_code):
    route = route_master_task({"query": query})
    assert route.mode == "fast_path"
    assert route.task_type == task_type
    assert route.target_agents == [agent]
    assert route.reason_code == reason_code


def test_multi_domain_request_never_uses_fast_path():
    query = "根据订单状态和退款规则帮我回复客户"
    with patch(
        "ecom_agent_matrix.modules.agent_cluster.master_router.is_llm_configured",
        return_value=True,
    ):
        route = route_master_task({"query": query})
    assert route.mode == "planner"
    assert route.reason_code == "AMBIGUOUS"
    assert route.target_agents == []


def test_unknown_route_depends_on_llm_availability():
    with patch(
        "ecom_agent_matrix.modules.agent_cluster.master_router.is_llm_configured",
        return_value=True,
    ):
        assert route_master_task({"query": "帮我分析一下这个问题"}).mode == "planner"
    with patch(
        "ecom_agent_matrix.modules.agent_cluster.master_router.is_llm_configured",
        return_value=False,
    ):
        route = route_master_task({"query": "帮我分析一下这个问题"})
    assert route.mode == "clarify" and route.reason_code == "UNKNOWN"


def test_fast_path_skips_planner_react_and_master_memory():
    async def scenario():
        request = MCPMessage(
            task_id="root-fast",
            correlation_id="gateway-correlation",
            sender="api_gateway",
            target=AGENT_MASTER,
            content={"task_type": "customer_service", "query": "帮我回复客户"},
        )
        memory = AsyncMock()
        observation = {
            "agent": AGENT_EXEC,
            "success": True,
            "data": {"answer": "已为您生成回复"},
            "error_msg": "",
            "timed_out": False,
        }
        with patch.object(master_module, "plan_sub_tasks_llm", new=AsyncMock()) as planner, patch.object(
            master_module, "react_decide", new=AsyncMock()
        ) as react, patch.object(
            master_module, "_react_call_one", new=AsyncMock(return_value=observation)
        ), patch.object(
            master_module, "polish_final_output", new=AsyncMock()
        ) as polish, patch.object(
            master_module.mcp_bus, "send_msg", new=AsyncMock(return_value=True)
        ) as send:
            await process_master_task(request, memory)
        return planner, react, polish, send, memory

    planner, react, polish, send, memory = asyncio.run(scenario())
    planner.assert_not_awaited()
    react.assert_not_awaited()
    polish.assert_not_awaited()
    memory.recall.assert_not_awaited()
    memory.safe_save_memory.assert_not_awaited()
    data = send.await_args.args[0].content["data"]
    assert data["master_llm_calls"] == {"planner": 0, "react": 0, "polish": 0, "total": 0}
    assert data["summary"] == "已为您生成回复"


def test_fast_path_uses_fresh_correlation_and_preserves_root_task_id():
    async def scenario():
        seen: list[tuple[str, str]] = []

        async def dispatch(task_id, correlation_id, target_agent, payload, priority):
            seen.append((task_id, correlation_id))
            child_request = MCPMessage(
                task_id=task_id,
                correlation_id=correlation_id,
                sender=AGENT_MASTER,
                target=target_agent,
                content=payload,
            )
            TaskReplyWaiter.submit_reply(
                build_reply(child_request, target_agent, success=True, data={"answer": "ok"})
            )

        route = MasterRouteDecision(
            mode="fast_path",
            task_type="knowledge_qa",
            target_agents=[AGENT_RAG],
            confidence=1,
            reason_code="EXPLICIT_TASK_TYPE",
        )
        with patch.object(master_module, "_dispatch_subtask", new=AsyncMock(side_effect=dispatch)):
            first = await execute_fast_path(
                MCPMessage(
                    task_id="same-root",
                    correlation_id="gateway-1",
                    sender="api_gateway",
                    target=AGENT_MASTER,
                    content={"query": "退款规则"},
                ),
                route,
            )
            second = await execute_fast_path(
                MCPMessage(
                    task_id="same-root",
                    correlation_id="gateway-2",
                    sender="api_gateway",
                    target=AGENT_MASTER,
                    content={"query": "退款规则"},
                ),
                route,
            )
        return first, second, seen

    first, second, seen = asyncio.run(scenario())
    assert first["task_id"] == second["task_id"] == "same-root"
    assert seen[0][0] == seen[1][0] == "same-root"
    assert seen[0][1] != seen[1][1]


@pytest.mark.parametrize(
    "agent,data,expected",
    [
        (AGENT_EXEC, {"answer": "CRM answer"}, "CRM answer"),
        (AGENT_EXEC, {"summary": "Report summary"}, "Report summary"),
        (AGENT_RAG, {"answer": "RAG answer"}, "RAG answer"),
    ],
)
def test_fast_path_reuses_child_text_without_polish(agent, data, expected):
    async def scenario():
        route = MasterRouteDecision(
            mode="fast_path",
            task_type="customer_service" if agent == AGENT_EXEC else "knowledge_qa",
            target_agents=[agent],
            confidence=1,
            reason_code="TEST",
        )
        with patch.object(
            master_module,
            "_react_call_one",
            new=AsyncMock(
                return_value={
                    "agent": agent,
                    "success": True,
                    "data": data,
                    "error_msg": "",
                    "timed_out": False,
                }
            ),
        ), patch.object(master_module, "polish_final_output", new=AsyncMock()) as polish:
            result = await execute_fast_path(
                MCPMessage(
                    task_id="root",
                    sender="api_gateway",
                    target=AGENT_MASTER,
                    content={"query": "test"},
                ),
                route,
            )
        return result, polish

    result, polish = asyncio.run(scenario())
    polish.assert_not_awaited()
    assert result["summary"] == expected
    assert result["master_llm_calls"]["total"] == 0


def test_react_immediate_finish_with_zero_observations_is_success():
    async def scenario():
        request = MCPMessage(
            task_id="root-zero",
            sender="api_gateway",
            target=AGENT_MASTER,
            content={"query": "complex ambiguous request"},
        )
        route = MasterRouteDecision(
            mode="planner", confidence=0.4, reason_code="AMBIGUOUS"
        )
        plan = PlanResult(
            sub_tasks=[], plan_confidence=0.9, reasoning="done", planner="test"
        )
        memory = AsyncMock()
        memory.recall.return_value = []
        with patch.object(master_module, "route_master_task", return_value=route), patch.object(
            master_module, "plan_sub_tasks_llm", new=AsyncMock(return_value=plan)
        ), patch.object(
            master_module,
            "react_decide",
            new=AsyncMock(
                return_value=ReactDecision(
                    thought="already complete", action="finish", final_answer="完成"
                )
            ),
        ), patch.object(master_module.mcp_bus, "send_msg", new=AsyncMock(return_value=True)) as send:
            await process_master_task(request, memory)
        return send.await_args_list[0].args[0].content["data"]

    data = asyncio.run(scenario())
    assert data["expected"] == 0 and data["received"] == 0
    assert data["timed_out"] is False
    assert data["all_success"] is True
    assert data["summary"] == "完成"


def test_react_real_timeout_remains_timeout():
    async def scenario():
        request = MCPMessage(
            task_id="root-timeout",
            sender="api_gateway",
            target=AGENT_MASTER,
            content={"query": "complex ambiguous request"},
        )
        route = MasterRouteDecision(
            mode="planner", confidence=0.4, reason_code="AMBIGUOUS"
        )
        plan = PlanResult(
            sub_tasks=[{"target_agent": AGENT_QUERY, "payload": {"query": "x"}}],
            plan_confidence=0.9,
            reasoning="query",
            planner="test",
        )
        memory = AsyncMock()
        memory.recall.return_value = []
        decision = ReactDecision(
            thought="query", action="call_agent", agent=AGENT_QUERY, payload={}
        )
        timeout_observation = {
            "agent": AGENT_QUERY,
            "success": False,
            "data": {},
            "error_msg": "timeout",
            "timed_out": True,
        }
        with patch.object(master_module, "route_master_task", return_value=route), patch.object(
            master_module, "plan_sub_tasks_llm", new=AsyncMock(return_value=plan)
        ), patch.object(
            master_module, "react_decide", new=AsyncMock(return_value=decision)
        ), patch.object(
            master_module, "_react_call_one", new=AsyncMock(return_value=timeout_observation)
        ), patch.object(
            master_module, "polish_final_output", new=AsyncMock(return_value="timeout")
        ), patch.object(master_module.mcp_bus, "send_msg", new=AsyncMock(return_value=True)) as send:
            await process_master_task(request, memory)
        return send.await_args_list[0].args[0].content["data"]

    data = asyncio.run(scenario())
    assert data["expected"] == 1 and data["received"] == 1
    assert data["timed_out"] is True
    assert data["all_success"] is False


def test_master_has_no_direct_skill_dependency_or_runtime():
    source = inspect.getsource(master_module)
    assert "exec_skill" not in source
    assert "modules.skills" not in source
    assert "_react_call_skill" not in source
    assert "call_skill" not in source


def test_complex_master_memory_is_compact_and_excludes_raw_payloads():
    async def scenario():
        request = MCPMessage(
            task_id="root-memory",
            sender="api_gateway",
            target=AGENT_MASTER,
            content={
                "query": "complex request",
                "api_token": "MUST_NOT_PERSIST",
                "history": ["large raw history"],
            },
        )
        route = MasterRouteDecision(
            mode="planner", confidence=0.4, reason_code="AMBIGUOUS"
        )
        plan = PlanResult(
            sub_tasks=[{"target_agent": AGENT_QUERY, "payload": {"query": "x"}}],
            plan_confidence=0.95,
            reasoning="short reason",
            planner="test",
        )
        decisions = [
            ReactDecision(
                thought="call", action="call_agent", agent=AGENT_QUERY, payload={}
            ),
            ReactDecision(thought="done", action="finish", final_answer="完成"),
        ]
        memory = AsyncMock()
        memory.recall.return_value = []
        observation = {
            "agent": AGENT_QUERY,
            "success": True,
            "data": {"rows": list(range(100))},
            "error_msg": "",
            "timed_out": False,
        }
        with patch.object(master_module, "route_master_task", return_value=route), patch.object(
            master_module, "plan_sub_tasks_llm", new=AsyncMock(return_value=plan)
        ), patch.object(
            master_module, "react_decide", new=AsyncMock(side_effect=decisions)
        ), patch.object(
            master_module, "_react_call_one", new=AsyncMock(return_value=observation)
        ), patch.object(master_module.mcp_bus, "send_msg", new=AsyncMock(return_value=True)):
            await process_master_task(request, memory)
        return memory.safe_save_memory.await_args.kwargs

    saved = asyncio.run(scenario())
    content = json.loads(saved["content"])
    assert set(content) == {"task_type", "route", "steps", "success", "latency_ms"}
    assert set(content["steps"][0]) == {"agent", "success", "reason_code"}
    serialized = json.dumps(content, ensure_ascii=False)
    assert "MUST_NOT_PERSIST" not in serialized
    assert "large raw history" not in serialized
    assert "rows" not in serialized
    assert saved["meta"]["agents"] == [AGENT_QUERY]
