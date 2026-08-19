"""Master Agent：Fast Path + Typed DAG planning + recovery orchestration。"""
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
from ecom_agent_matrix.core.security import (
    authorize_task,
    authorize_task_types,
    security_log_fields,
    require_trusted_ingress,
)
from ecom_agent_matrix.core.security.errors import AuthorizationError
from ecom_agent_matrix.modules.agent_cluster.master.executor import MasterPlanExecutor
from ecom_agent_matrix.modules.agent_cluster.master.planner import typed_master_planner
from ecom_agent_matrix.modules.agent_cluster.master.react import recovery_controller
from ecom_agent_matrix.modules.agent_cluster.master.recovery import apply_recovery_decision
from ecom_agent_matrix.modules.agent_cluster.master.schemas import PlanExecutionResult
from ecom_agent_matrix.modules.agent_cluster.master.telemetry import MasterLLMTelemetry
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
    security=None,
    approval=None,
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
            security=security,
            approval=approval,
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
    security=None,
    approval=None,
) -> dict:
    """ReAct 单步：下发一个 Agent → 等待回传 → 返回 observation。"""
    correlation_id = str(uuid.uuid4())
    TaskReplyWaiter.begin(correlation_id, 1)
    try:
        if security is None and approval is None:
            await _dispatch_subtask(
                task_id, correlation_id, target_agent, payload, priority
            )
        elif approval is None:
            await _dispatch_subtask(
                task_id, correlation_id, target_agent, payload, priority, security
            )
        else:
            await _dispatch_subtask(
                task_id, correlation_id, target_agent, payload, priority, security, approval
            )
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


def _compat_llm_calls(usage: dict[str, Any]) -> dict[str, int]:
    """保留 Phase 3A 字段；数据源仅来自真实 Provider telemetry。"""
    return {
        "planner": int(usage.get("planner", {}).get("calls", 0)),
        "react": int(usage.get("recovery", {}).get("calls", 0)),
        "polish": int(usage.get("polish", {}).get("calls", 0)),
        "total": int(usage.get("calls", 0)),
    }


def _execution_summary(execution: PlanExecutionResult) -> str:
    """优先返回最终业务步骤的自然语言结果，避免无意义 polish。"""
    terminal = list(execution.step_results.values())
    for result in reversed(terminal):
        if result.status == "SUCCESS":
            summary = _existing_summary(result.data)
            if summary:
                return summary
    success_count = sum(item.status == "SUCCESS" for item in terminal)
    if execution.all_success:
        return f"任务已完成，{success_count}/{len(terminal)} 个步骤成功。"
    return f"任务部分完成，{success_count}/{len(terminal)} 个步骤成功。"


async def _save_plan_memory(
    long_mem: AgentLongVectorMemory,
    *,
    task_type: str,
    plan: Any,
    execution: PlanExecutionResult,
    usage: dict[str, Any],
    recovery_actions: list[dict[str, Any]] | None = None,
    security=None,
) -> None:
    """Complex plan 仅保存紧凑状态，不持久化 payload/result/reasoning。"""
    compact = {
        "task_type": task_type,
        "plan": {
            "reason_code": plan.reason_code,
            "confidence": plan.confidence,
            "source": plan.planner_source,
            "steps": [
                {
                    "step_id": step.step_id,
                    "agent": step.agent,
                    "task_type": step.task_type,
                    "depends_on": step.depends_on,
                }
                for step in plan.steps
            ],
        },
        "step_statuses": {
            step_id: {
                "status": result.status,
                "error_code": result.error_code,
                "latency_ms": result.latency_ms,
            }
            for step_id, result in execution.step_results.items()
        },
        "usage": usage,
        "recovery": {
            "actions": list(recovery_actions or []),
            "final_status": "SUCCESS" if execution.all_success else "FAILED",
        },
    }
    await long_mem.safe_save_memory(
        agent_name=AGENT_MASTER,
        content=json.dumps(compact, ensure_ascii=False),
        meta={
            "task_type": task_type,
            "mode": "plan",
            "plan_confidence": plan.confidence,
            "success": execution.all_success,
            "verified": False,
            "deprecated": False,
        },
        context=security,
    )


async def _process_complex_plan(
    msg: MCPMessage,
    long_mem: AgentLongVectorMemory,
    route: MasterRouteDecision,
    task_input: dict[str, Any],
    started: float,
) -> None:
    telemetry = MasterLLMTelemetry()
    plan = await typed_master_planner.plan(task_input, telemetry)
    if plan.decision == "clarify":
        usage = telemetry.snapshot().model_dump()
        calls = _compat_llm_calls(usage)
        result = {
            "task_id": msg.task_id,
            "mode": "clarify",
            "expected": 0,
            "received": 0,
            "timed_out": False,
            "all_success": True,
            "partial_success": False,
            "step_results": {},
            "summary": plan.clarification_question,
            "plan": plan.model_dump(),
            "route": route.model_dump(),
            "master_llm_usage": usage,
            "master_llm_calls": calls,
            "metadata": {
                "master_llm_usage": usage,
                "master_llm_calls": calls,
                "master_memory": "skipped_clarify",
            },
        }
        await mcp_bus.send_msg(
            build_reply(
                msg,
                sender=AGENT_MASTER,
                success=True,
                data=result,
                msg_type="master_task_result",
            )
        )
        return

    if msg.security is not None:
        try:
            authorize_task_types(msg.security, (step.task_type for step in plan.steps))
        except AuthorizationError as exc:
            await mcp_bus.send_msg(
                build_reply(
                    msg,
                    sender=AGENT_MASTER,
                    success=False,
                    data={"error_code": exc.error_code, "task_type": exc.task_type},
                    error_msg="PERMISSION_DENIED",
                    msg_type="master_task_result",
                )
            )
            return

    executor = MasterPlanExecutor()
    execution = await executor.execute(plan, msg)
    recovery_actions: list[dict[str, Any]] = []
    terminal_recovery = None
    for _ in range(max(0, int(settings.MASTER_RECOVERY_MAX_STEPS))):
        if execution.all_success:
            break
        decision = await recovery_controller.run(execution, telemetry)
        if decision is None:
            break
        applied = await apply_recovery_decision(
            decision,
            plan=plan,
            execution=execution,
            root_message=msg,
            task_input=task_input,
            executor=executor,
            planner=typed_master_planner,
            telemetry=telemetry,
        )
        plan = applied.plan
        execution = applied.execution
        terminal_recovery = applied.decision
        recovery_actions.append(
            {
                "action": applied.decision.action,
                "step_id": applied.decision.step_id,
                "reason_code": applied.decision.reason_code,
            }
        )
        if applied.decision.action in {"finish", "clarify"}:
            break
        if not applied.continue_recovery:
            break

    summary = ""
    if terminal_recovery is not None:
        summary = (
            terminal_recovery.final_answer
            or terminal_recovery.clarification_question
        )
    if not summary:
        summary = _execution_summary(execution)

    usage = telemetry.snapshot().model_dump()
    calls = _compat_llm_calls(usage)
    step_results = {
        step_id: item.model_dump()
        for step_id, item in execution.step_results.items()
    }
    result = {
        "task_id": msg.task_id,
        "mode": "plan",
        "expected": len(plan.steps),
        "received": sum(
            item.status in {"SUCCESS", "FAILED"}
            for item in execution.step_results.values()
        ),
        "timed_out": execution.timed_out,
        "all_success": execution.all_success,
        "partial_success": execution.partial_success,
        "step_results": step_results,
        "sub_results": list(step_results.values()),
        "summary": summary,
        "plan": plan.model_dump(),
        "route": route.model_dump(),
        "recovery": {
            "attempted": bool(recovery_actions),
            "actions": recovery_actions,
            "final_status": "SUCCESS" if execution.all_success else "FAILED",
        },
        "master_llm_usage": usage,
        "master_llm_calls": calls,
        "metadata": {
            "master_llm_usage": usage,
            "master_llm_calls": calls,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    }
    await mcp_bus.send_msg(
        build_reply(
            msg,
            sender=AGENT_MASTER,
            success=execution.all_success,
            data=result,
            error_msg="" if execution.all_success else "部分计划步骤未成功",
            msg_type="master_task_result",
        )
    )
    try:
        await _save_plan_memory(
            long_mem,
            task_type=str(task_input.get("task_type") or "unknown"),
            plan=plan,
            execution=execution,
            usage=usage,
            recovery_actions=recovery_actions,
            security=msg.security,
        )
    except Exception as exc:
        logger.warning(
            "master_memory_save_failed",
            extra={
                "event": "master_memory_save_failed",
                "task_id": msg.task_id,
                "error_type": type(exc).__name__,
            },
        )


async def execute_fast_path(
    msg: MCPMessage,
    route: MasterRouteDecision,
) -> dict[str, Any]:
    """单次 Agent dispatch；不进入 Planner、ReAct 或 Master Memory。"""
    started = time.perf_counter()
    telemetry = MasterLLMTelemetry()
    target_agent = route.target_agents[0]
    payload = {**dict(msg.content or {}), "task_type": route.task_type}
    if route.task_type == "goods_catalog":
        payload["mode"] = "catalog"

    observation = await _react_call_one(
        msg.task_id,
        target_agent,
        payload,
        msg.priority,
        msg.security,
        msg.approval,
    )
    timed_out = bool(observation.get("timed_out"))
    success = bool(observation.get("success")) and not timed_out
    summary = _existing_summary(observation.get("data") or {})
    if not summary:
        summary = await polish_final_output(
            success=success,
            data=observation.get("data") or {},
            error_msg=observation.get("error_msg", ""),
            user_query=str(msg.content.get("query") or msg.content.get("user_query") or ""),
            reply_from=AGENT_MASTER,
            prefer_existing_answer=True,
            on_provider_start=lambda: telemetry.start_call("polish"),
            on_provider_result=lambda result: telemetry.add_result("polish", result),
        )

    usage = telemetry.snapshot().model_dump()
    calls = _compat_llm_calls(usage)
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
        "master_llm_usage": usage,
        "metadata": {
            "master_llm_calls": calls,
            "master_llm_usage": usage,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "master_memory": "skipped_fast_path",
        },
    }


async def process_master_task(msg: MCPMessage, long_mem: AgentLongVectorMemory) -> None:
    """Route to Fast Path or validated DAG, then apply bounded recovery if needed."""
    task_id = msg.task_id
    require_trusted_ingress(msg.security, app_env=settings.APP_ENV)
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
            **security_log_fields(msg.security),
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

    if route.task_type and msg.security is not None:
        try:
            authorize_task(msg.security, route.task_type)
        except AuthorizationError as exc:
            await mcp_bus.send_msg(
                build_reply(
                    msg,
                    sender=AGENT_MASTER,
                    success=False,
                    data={"error_code": exc.error_code, "task_type": exc.task_type},
                    error_msg="PERMISSION_DENIED",
                    msg_type="master_task_result",
                )
            )
            return

    if route.mode == "clarify":
        clarification = "请说明您要查询的数据、咨询的店铺规则，或需要执行的业务操作。"
        usage = MasterLLMTelemetry().snapshot().model_dump()
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
            "master_llm_usage": usage,
            "master_llm_calls": calls,
            "metadata": {
                "master_llm_usage": usage,
                "master_llm_calls": calls,
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

    await _process_complex_plan(msg, long_mem, route, task_input, started)
    return

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
            "master_llm_usage": MasterLLMTelemetry().snapshot().model_dump(),
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
