"""总控规划 Agent（ReAct 多轮编排）。"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any

from ecom_agent_matrix.config.constants import AGENT_MASTER
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.memory.long_vector_memory import AgentLongVectorMemory
from ecom_agent_matrix.core.mcp.bus import mcp_bus
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.registry import register_agent
from ecom_agent_matrix.core.mcp.reply import build_reply
from ecom_agent_matrix.core.mcp.task_waiter import TaskReplyWaiter, is_agent_reply
from ecom_agent_matrix.core.llm.output_polish import polish_final_output
from ecom_agent_matrix.modules.agent_cluster.master_planner import (
    merge_observation_into_working,
    plan_sub_tasks_llm,
    react_decide,
    should_save_to_memory,
)
from ecom_agent_matrix.modules.agent_cluster.master_router import (
    MasterRouteDecision,
    route_master_task,
)

logger = setup_logger("agent.master")

_subtask_semaphore: asyncio.Semaphore | None = None
_master_task_semaphore: asyncio.Semaphore | None = None
_master_tasks: set[asyncio.Task] = set()


def _get_subtask_semaphore() -> asyncio.Semaphore:
    global _subtask_semaphore
    if _subtask_semaphore is None:
        _subtask_semaphore = asyncio.Semaphore(settings.MASTER_MAX_SUBTASK_CONCURRENT)
    return _subtask_semaphore


def _get_master_task_semaphore() -> asyncio.Semaphore:
    global _master_task_semaphore
    if _master_task_semaphore is None:
        _master_task_semaphore = asyncio.Semaphore(int(settings.MASTER_MAX_CONCURRENT))
    return _master_task_semaphore


def aggregate_sub_replies(task_id: str, replies: list[MCPMessage], expected: int) -> dict[str, Any]:
    """聚合子 Agent 回传结果。"""
    sub_results = []
    for msg in replies:
        sub_results.append(
            {
                "agent": msg.sender,
                "success": msg.content.get("success", False),
                "data": msg.content.get("data", {}),
                "error_msg": msg.content.get("error_msg", ""),
            }
        )
    return {
        "task_id": task_id,
        "expected": expected,
        "received": len(replies),
        "timed_out": len(replies) < expected,
        "all_success": len(replies) == expected and all(r["success"] for r in sub_results),
        "sub_results": sub_results,
    }


def _observation_from_reply(reply: MCPMessage | None, target_agent: str, timed_out: bool) -> dict:
    if reply is None:
        return {
            "agent": target_agent,
            "success": False,
            "data": {},
            "error_msg": "子任务超时无回传",
            "timed_out": True,
        }
    return {
        "agent": reply.sender or target_agent,
        "success": bool(reply.content.get("success")) and not timed_out,
        "data": reply.content.get("data") or {},
        "error_msg": reply.content.get("error_msg", ""),
        "timed_out": timed_out,
    }


async def _dispatch_subtask(
    task_id: str,
    correlation_id: str,
    target_agent: str,
    payload: dict,
    priority: int,
) -> None:
    """限流后向子 Agent 下发单步任务。"""
    sem = _get_subtask_semaphore()
    async with sem:
        # 去掉过大的内部字段；Master 不写业务解析
        clean = {k: v for k, v in payload.items() if k not in ("_memory_context",)}
        sub_msg = MCPMessage(
            task_id=task_id,
            correlation_id=correlation_id,
            sender=AGENT_MASTER,
            target=target_agent,
            priority=priority,
            content=clean,
        )
        await mcp_bus.send_msg(sub_msg)
        logger.info(
            "subtask_dispatched",
            extra={
                "event": "subtask_dispatched",
                "task_id": task_id,
                "correlation_id": correlation_id,
                "agent": target_agent,
            },
        )


async def _react_call_one(
    task_id: str,
    target_agent: str,
    payload: dict,
    priority: int,
) -> dict:
    """ReAct 单步：下发一个 Agent → 等待回传 → 返回 observation。"""
    correlation_id = str(uuid.uuid4())
    TaskReplyWaiter.begin(correlation_id, 1)
    try:
        await _dispatch_subtask(task_id, correlation_id, target_agent, payload, priority)
        replies = await TaskReplyWaiter.wait(correlation_id, timeout=float(settings.MCP_TIMEOUT))
    finally:
        TaskReplyWaiter.discard(correlation_id)
    timed_out = len(replies) < 1
    reply = replies[0] if replies else None
    return _observation_from_reply(reply, target_agent, timed_out)


def _existing_summary(data: dict) -> str:
    """优先复用子 Agent 已生成的面向用户文本。"""
    for key in ("answer", "summary", "advice"):
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _llm_call_metadata(*, planner: int = 0, react: int = 0, polish: int = 0) -> dict:
    return {
        "planner": planner,
        "react": react,
        "polish": polish,
        "total": planner + react + polish,
    }


def _safe_task_descriptor(task_input: dict) -> dict[str, Any]:
    """仅保留路由/召回必要字段，避免把完整 payload 送入日志或 Memory。"""
    return {
        "task_type": str(task_input.get("task_type") or "unknown")[:80],
        "query": _redact_sensitive_text(
            str(task_input.get("query") or task_input.get("user_query") or "")
        )[:500],
        "sku": str(task_input.get("sku") or task_input.get("target_sku") or "")[:120],
    }


def _redact_sensitive_text(text: str) -> str:
    value = re.sub(
        r"(?i)\b(api[_-]?key|token|password|secret)\b\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        str(text or ""),
    )
    return re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+", "Bearer [REDACTED]", value)


def _react_reason_code(action: str) -> str:
    if action == "call_agent":
        return "REACT_CALL_AGENT"
    if action == "finish":
        return "REACT_FINISH"
    return "REACT_INVALID_ACTION"


async def execute_fast_path(
    msg: MCPMessage,
    route: MasterRouteDecision,
) -> dict[str, Any]:
    """单次 Agent dispatch；不进入 Planner、ReAct 或 Master Memory。"""
    started = time.perf_counter()
    target_agent = route.target_agents[0]
    payload = {**dict(msg.content or {}), "task_type": route.task_type}
    if route.task_type == "goods_catalog":
        payload["mode"] = "catalog"

    observation = await _react_call_one(
        msg.task_id,
        target_agent,
        payload,
        msg.priority,
    )
    timed_out = bool(observation.get("timed_out"))
    success = bool(observation.get("success")) and not timed_out
    summary = _existing_summary(observation.get("data") or {})
    polish_calls = 0
    if not summary:
        polish_calls = 1
        summary = await polish_final_output(
            success=success,
            data=observation.get("data") or {},
            error_msg=observation.get("error_msg", ""),
            user_query=str(msg.content.get("query") or msg.content.get("user_query") or ""),
            reply_from=AGENT_MASTER,
            prefer_existing_answer=True,
        )

    calls = _llm_call_metadata(polish=polish_calls)
    return {
        "task_id": msg.task_id,
        "mode": "fast_path",
        "expected": 1,
        "received": 1 if not timed_out else 0,
        "timed_out": timed_out,
        "all_success": success,
        "sub_results": [
            {
                "agent": observation.get("agent") or target_agent,
                "success": success,
                "data": observation.get("data") or {},
                "error_msg": observation.get("error_msg", ""),
            }
        ],
        "summary": summary,
        "route": route.model_dump(),
        "master_llm_calls": calls,
        "metadata": {
            "master_llm_calls": calls,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "master_memory": "skipped_fast_path",
        },
    }


async def process_master_task(msg: MCPMessage, long_mem: AgentLongVectorMemory) -> None:
    """
    ReAct 主流程：
    召回 → 初始规划（建议序列）→ 逐步 Thought/Action/Observation → 聚合回传 → 条件记忆。
    子 Agent 仅 Query / Exec / RAG；SKU 解析在 Query 内部完成。
    """
    task_id = msg.task_id
    started = time.perf_counter()
    task_input = dict(msg.content or {})
    task_type = task_input.get("task_type", "unknown")
    task_descriptor = _safe_task_descriptor(task_input)

    logger.info(
        "master_task_received",
        extra={
            "event": "master_task_received",
            "task_id": task_id,
            "task_type": task_descriptor["task_type"],
            "query": task_descriptor["query"][:200],
        },
    )

    route = route_master_task(task_input)
    logger.info(
        "master_route_done",
        extra={
            "event": "master_route_done",
            "task_id": task_id,
            "mode": route.mode,
            "task_type": route.task_type or "",
            "reason_code": route.reason_code,
            "confidence": route.confidence,
        },
    )

    if route.mode == "clarify":
        clarification = "请说明您要查询的数据、咨询的店铺规则，或需要执行的业务操作。"
        calls = _llm_call_metadata()
        final_result = {
            "task_id": task_id,
            "mode": "clarify",
            "expected": 0,
            "received": 0,
            "timed_out": False,
            "all_success": True,
            "sub_results": [],
            "react_trace": [],
            "summary": clarification,
            "route": route.model_dump(),
            "master_llm_calls": calls,
            "metadata": {"master_llm_calls": calls, "master_memory": "skipped_clarify"},
        }
        await mcp_bus.send_msg(
            build_reply(
                msg,
                sender=AGENT_MASTER,
                success=True,
                data=final_result,
                msg_type="master_task_result",
            )
        )
        return

    if route.mode == "fast_path":
        final_result = await execute_fast_path(msg, route)
        await mcp_bus.send_msg(
            build_reply(
                msg,
                sender=AGENT_MASTER,
                success=final_result["all_success"] and not final_result["timed_out"],
                data=final_result,
                error_msg=(
                    "子任务超时或未成功"
                    if final_result["timed_out"] or not final_result["all_success"]
                    else ""
                ),
                msg_type="master_task_result",
            )
        )
        return

    memory_hits: list[dict] = []
    try:
        memory_hits = await long_mem.recall(
            query_text=json.dumps(task_descriptor, ensure_ascii=False),
            agent_name=AGENT_MASTER,
            top_k=2,
        )
    except Exception as exc:
        logger.warning(
            "master_memory_recall_failed",
            extra={
                "event": "master_memory_recall_failed",
                "task_id": task_id,
                "error_type": type(exc).__name__,
            },
        )

    planner_calls = 1
    react_calls = 0
    polish_calls = 0
    plan = await plan_sub_tasks_llm(task_input, memory_hits)
    suggested_agents = [s["target_agent"] for s in plan.sub_tasks]

    logger.info(
        "master_plan_done",
        extra={
            "event": "master_plan_done",
            "task_id": task_id,
            "planner": plan.planner,
            "plan_confidence": plan.plan_confidence,
            "agents": suggested_agents,
        },
    )

    if plan.decision == "clarify":
        clarification = plan.clarification_question or "请补充您想查询或处理的具体业务内容。"
        final_result = {
            "task_id": task_id,
            "mode": "clarify",
            "expected": 0,
            "received": 0,
            "timed_out": False,
            "all_success": True,
            "sub_results": [],
            "react_trace": [],
            "summary": clarification,
            "plan": {
                "decision": "clarify",
                "planner": plan.planner,
                "plan_confidence": plan.plan_confidence,
                "reasoning": plan.reasoning,
                "agents": [],
                "reason_code": route.reason_code,
            },
            "route": route.model_dump(),
            "master_llm_calls": _llm_call_metadata(planner=planner_calls),
            "metadata": {
                "master_llm_calls": _llm_call_metadata(planner=planner_calls),
                "master_memory": "skipped_clarify",
            },
        }
        await mcp_bus.send_msg(
            build_reply(
                msg,
                sender=AGENT_MASTER,
                success=True,
                data=final_result,
                msg_type="master_task_result",
            )
        )
        logger.info(
            "master_task_clarify",
            extra={"event": "master_task_clarify", "task_id": task_id},
        )
        return

    working = dict(plan.sub_tasks[0]["payload"]) if plan.sub_tasks else dict(task_input)
    observations: list[dict] = []
    react_trace: list[dict] = []
    max_steps = int(settings.MASTER_REACT_MAX_STEPS)
    finished_normally = False
    terminal_error = False
    final_answer = ""

    for step in range(1, max_steps + 1):
        react_calls += 1
        decision = await react_decide(working, observations, suggested_agents)
        react_trace.append(
            {
                "step": step,
                "reason_code": _react_reason_code(decision.action),
                "confidence": decision.confidence,
                "action": decision.action,
                "agent": decision.agent,
                "skill": decision.skill,
                "source": decision.source,
            }
        )
        logger.info(
            "master_react_step",
            extra={
                "event": "master_react_step",
                "task_id": task_id,
                "reason_code": _react_reason_code(decision.action),
                "query": (
                    f"step={step} action={decision.action} "
                    f"agent={decision.agent} skill={decision.skill}"
                ),
            },
        )

        if decision.action == "finish":
            react_trace[-1]["result"] = {"final_answer": decision.final_answer}
            finished_normally = True
            final_answer = str(decision.final_answer or "").strip()
            break

        if decision.action != "call_agent" or not decision.agent:
            react_trace[-1]["result"] = {
                "success": False,
                "error_msg": "非法 ReAct 动作，提前结束",
            }
            terminal_error = True
            break

        obs = await _react_call_one(task_id, decision.agent, decision.payload, msg.priority)
        observations.append(obs)
        working = merge_observation_into_working(working, obs)
        react_trace[-1]["result"] = {
            "success": obs.get("success"),
            "error_msg": obs.get("error_msg", ""),
        }

        # 关键失败且后续依赖该结果时，规则层会在下一步 finish；此处也可提前跳出
        if obs.get("timed_out"):
            break

    else:
        finished_normally = True
        final_answer = "达到最大步数，强制结束"
        react_trace.append(
            {
                "step": max_steps + 1,
                "decision_reason": "达到 ReAct 最大步数",
                "confidence": 1.0,
                "action": "finish",
                "agent": "",
                "skill": "",
                "source": "rules",
                "result": {"final_answer": "达到最大步数，强制结束"},
            }
        )

    expected = len(observations)
    timed_out = any(o.get("timed_out") for o in observations)
    all_success = (
        finished_normally
        and not terminal_error
        and not timed_out
        and all(o.get("success") for o in observations)
    )
    final_result = {
        "task_id": task_id,
        "mode": "react",
        "expected": expected,
        "received": len(observations),
        "timed_out": timed_out,
        "all_success": all_success,
        "sub_results": [
            {
                "agent": o.get("agent"),
                "success": o.get("success", False),
                "data": o.get("data", {}),
                "error_msg": o.get("error_msg", ""),
            }
            for o in observations
        ],
        "react_trace": react_trace,
        "working_sku": working.get("sku"),
        "plan": {
            "decision": plan.decision,
            "planner": plan.planner,
            "plan_confidence": plan.plan_confidence,
            "reasoning": plan.reasoning,
            "agents": suggested_agents,
            "reason_code": route.reason_code,
        },
        "route": route.model_dump(),
    }
    elapsed_ms = (time.perf_counter() - started) * 1000

    summary = final_answer
    if not summary and observations:
        summary = _existing_summary(observations[-1].get("data") or {})
    if not summary:
        polish_calls = 1
        summary = await polish_final_output(
            success=final_result["all_success"] and not final_result["timed_out"],
            data=final_result,
            error_msg=(
                "部分子任务超时或未成功"
                if final_result["timed_out"] or not final_result["all_success"]
                else ""
            ),
            user_query=str(task_input.get("query") or task_input.get("user_query") or ""),
            reply_from=AGENT_MASTER,
            prefer_existing_answer=True,
        )
    final_result["summary"] = summary
    calls = _llm_call_metadata(
        planner=planner_calls,
        react=react_calls,
        polish=polish_calls,
    )
    final_result["master_llm_calls"] = calls
    final_result["metadata"] = {"master_llm_calls": calls}

    reply = build_reply(
        msg,
        sender=AGENT_MASTER,
        success=final_result["all_success"] and not final_result["timed_out"],
        data=final_result,
        error_msg=(
            "部分子任务超时或未成功"
            if final_result["timed_out"] or not final_result["all_success"]
            else ""
        ),
        msg_type="master_task_result",
    )
    await mcp_bus.send_msg(reply)

    save_ok, mem_confidence = should_save_to_memory(plan.plan_confidence, final_result)
    if save_ok:
        memory_task_type = (
            task_type
            if task_type != "unknown"
            else working.get("_inferred_task_type") or working.get("task_type") or "unknown"
        )
        await long_mem.safe_save_memory(
            agent_name=AGENT_MASTER,
            content=json.dumps(
                {
                    "task_type": memory_task_type,
                    "route": suggested_agents,
                    "steps": [
                        {
                            "agent": item.get("agent"),
                            "success": bool(item.get("success")),
                            "reason_code": (
                                "AGENT_SUCCESS" if item.get("success") else "AGENT_FAILED"
                            ),
                        }
                        for item in observations
                    ],
                    "success": final_result["all_success"],
                    "latency_ms": round(elapsed_ms, 2),
                },
                ensure_ascii=False,
            ),
            meta={
                "task_type": memory_task_type,
                "planner": plan.planner,
                "plan_confidence": plan.plan_confidence,
                "confidence": mem_confidence,
                "memory_hits_used": len(memory_hits),
                "latency_ms": round(elapsed_ms, 2),
                "mode": "react",
                "agents": suggested_agents,
                "sku": working.get("sku"),
                "success": final_result["all_success"],
                "verified": False,
                "deprecated": False,
            },
        )
        logger.info(
            "master_memory_saved",
            extra={
                "event": "master_memory_saved",
                "task_id": task_id,
                "confidence": mem_confidence,
            },
        )
    else:
        logger.info(
            "master_memory_skipped",
            extra={
                "event": "master_memory_skipped",
                "task_id": task_id,
                "plan_confidence": plan.plan_confidence,
                "memory_confidence": mem_confidence,
                "all_success": final_result["all_success"],
                "timed_out": final_result["timed_out"],
            },
        )

    logger.info(
        "master_task_done",
        extra={
            "event": "master_task_done",
            "task_id": task_id,
            "recall_count": final_result["received"],
            "latency_ms": round(elapsed_ms, 2),
        },
    )


async def safe_process_master_task(
    msg: MCPMessage,
    long_mem: AgentLongVectorMemory,
) -> None:
    """限制用户级并发，并保证未捕获异常也向 Gateway 回传。"""
    try:
        async with _get_master_task_semaphore():
            await process_master_task(msg, long_mem)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception(
            "master_task_failed",
            extra={
                "event": "master_task_failed",
                "task_id": msg.task_id,
                "error_type": type(exc).__name__,
            },
        )
        failure = {
            "task_id": msg.task_id,
            "mode": "master_error",
            "expected": 0,
            "received": 0,
            "timed_out": False,
            "all_success": False,
            "sub_results": [],
            "summary": "master task failed",
            "master_llm_calls": _llm_call_metadata(),
            "metadata": {"master_llm_calls": _llm_call_metadata()},
        }
        try:
            await mcp_bus.send_msg(
                build_reply(
                    msg,
                    sender=AGENT_MASTER,
                    success=False,
                    data=failure,
                    error_msg="master task failed",
                    msg_type="master_task_result",
                )
            )
        except Exception as reply_exc:
            logger.error(
                "master_failure_reply_failed",
                extra={
                    "event": "master_failure_reply_failed",
                    "task_id": msg.task_id,
                    "error_type": type(reply_exc).__name__,
                },
            )


def _consume_master_task(task: asyncio.Task) -> None:
    """移除并消费后台任务结果，避免未检索异常告警。"""
    _master_tasks.discard(task)
    if task.cancelled():
        return
    try:
        error = task.exception()
    except asyncio.CancelledError:
        return
    if error is not None:
        logger.error(
            "master_background_task_error",
            extra={
                "event": "master_background_task_error",
                "error_type": type(error).__name__,
            },
        )


def _track_master_task(
    msg: MCPMessage,
    long_mem: AgentLongVectorMemory,
) -> asyncio.Task:
    task = asyncio.create_task(safe_process_master_task(msg, long_mem))
    _master_tasks.add(task)
    task.add_done_callback(_consume_master_task)
    return task


@register_agent(AGENT_MASTER)
async def master_agent(msg_queue: asyncio.Queue):
    """总控 Master Agent 主循环。"""
    logger.info("master_agent_started", extra={"event": "master_agent_started", "agent": AGENT_MASTER})
    long_mem = AgentLongVectorMemory()

    while True:
        msg: MCPMessage = await msg_queue.get()
        try:
            if is_agent_reply(msg):
                TaskReplyWaiter.submit_reply(msg)
                continue
            _track_master_task(msg, long_mem)
        except Exception as exc:
            logger.exception(
                "master_loop_error",
                extra={
                    "event": "master_loop_error",
                    "task_id": msg.task_id,
                    "error_type": type(exc).__name__,
                },
            )
        finally:
            msg_queue.task_done()
