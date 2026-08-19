"""Validated MasterPlan 的 deterministic DAG executor。"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from ecom_agent_matrix.config.constants import AGENT_MASTER
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.mcp.bus import mcp_bus
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.task_waiter import TaskReplyWaiter
from ecom_agent_matrix.modules.agent_cluster.master.policy import validate_master_plan
from ecom_agent_matrix.modules.agent_cluster.master.schemas import (
    MasterPlan,
    PlanExecutionResult,
    PlanStep,
    StepResult,
)

DEPENDENCY_FAILED = "DEPENDENCY_FAILED"
STEP_TIMEOUT = "TIMEOUT"
AGENT_FAILED = "AGENT_FAILED"
STEP_EXECUTION_ERROR = "STEP_EXECUTION_ERROR"
_TERMINAL = frozenset({"SUCCESS", "FAILED", "SKIPPED"})
_FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "reasoning_content",
        "react_trace",
        "memory_context",
        "_memory_context",
        "logs",
        "history",
        "api_key",
        "token",
        "password",
        "secret",
    }
)
_UPSTREAM_MAX_CHARS = 2500
logger = setup_logger("agent.master.executor")


def _sanitize_context(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_context(item)
            for key, item in value.items()
            if str(key).lower() not in _FORBIDDEN_CONTEXT_KEYS
            and not any(
                marker in str(key).lower()
                for marker in ("password", "secret", "token", "api_key", "reasoning")
            )
        }
    if isinstance(value, list):
        return [_sanitize_context(item) for item in value[:50]]
    if isinstance(value, tuple):
        return [_sanitize_context(item) for item in value[:50]]
    if isinstance(value, str):
        return value[:1200]
    return value


def _compact_step_data(data: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_context(data)
    serialized = json.dumps(sanitized, ensure_ascii=False, default=str)
    if len(serialized) <= _UPSTREAM_MAX_CHARS:
        return sanitized
    for key in ("answer", "summary", "advice", "order_no", "status", "query_kind"):
        if sanitized.get(key) not in (None, "", [], {}):
            return {key: sanitized[key], "truncated": True}
    return {"preview": serialized[: _UPSTREAM_MAX_CHARS - 40], "truncated": True}


class MasterPlanExecutor:
    def __init__(self, *, max_concurrent: int | None = None, timeout: float | None = None):
        limit = int(
            max_concurrent
            if max_concurrent is not None
            else settings.MASTER_MAX_SUBTASK_CONCURRENT
        )
        self._semaphore = asyncio.Semaphore(max(limit, 1))
        self.timeout = float(timeout if timeout is not None else settings.MCP_TIMEOUT)

    async def execute(
        self,
        plan: MasterPlan,
        root_message: MCPMessage,
    ) -> PlanExecutionResult:
        validate_master_plan(plan)
        started = time.perf_counter()
        steps = {step.step_id: step for step in plan.steps}
        results = {
            step.step_id: StepResult(
                step_id=step.step_id,
                agent=step.agent,
                task_type=step.task_type,
                status="PENDING",
            )
            for step in plan.steps
        }

        while any(result.status == "PENDING" for result in results.values()):
            state_changed = False
            for step in plan.steps:
                result = results[step.step_id]
                if result.status != "PENDING":
                    continue
                failed_required = [
                    dependency
                    for dependency in step.depends_on
                    if results[dependency].status in {"FAILED", "SKIPPED"}
                    and steps[dependency].required
                ]
                if failed_required:
                    state_changed = True
                    results[step.step_id] = result.model_copy(
                        update={
                            "status": "SKIPPED",
                            "success": False,
                            "error_code": DEPENDENCY_FAILED,
                            "error_msg": (
                                "required dependencies failed: " + ", ".join(failed_required)
                            ),
                        }
                    )

            ready = [
                step
                for step in plan.steps
                if results[step.step_id].status == "PENDING"
                and all(results[dep].status in _TERMINAL for dep in step.depends_on)
            ]
            if not ready:
                if any(result.status == "PENDING" for result in results.values()):
                    if state_changed:
                        continue
                    raise RuntimeError("validated plan has no executable ready step")
                break

            for step in ready:
                results[step.step_id] = results[step.step_id].model_copy(
                    update={"status": "RUNNING"}
                )
            completed = await asyncio.gather(
                *(
                    self._execute_step(
                        step,
                        root_message,
                        self._upstream_context(step, steps, results),
                    )
                    for step in ready
                )
            )
            for result in completed:
                results[result.step_id] = result

        success_count = sum(result.status == "SUCCESS" for result in results.values())
        all_success = bool(results) and success_count == len(results)
        partial = 0 < success_count < len(results)
        timed_out = any(result.error_code == STEP_TIMEOUT for result in results.values())
        return PlanExecutionResult(
            step_results=results,
            all_success=all_success,
            partial_success=partial,
            timed_out=timed_out,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def _upstream_context(
        self,
        step: PlanStep,
        steps: dict[str, PlanStep],
        results: dict[str, StepResult],
    ) -> dict[str, Any]:
        return {
            dependency: {
                "agent": steps[dependency].agent,
                "task_type": steps[dependency].task_type,
                "data": _compact_step_data(results[dependency].data),
            }
            for dependency in step.depends_on
            if results[dependency].status == "SUCCESS"
        }

    async def _execute_step(
        self,
        step: PlanStep,
        root_message: MCPMessage,
        upstream_context: dict[str, Any],
    ) -> StepResult:
        async with self._semaphore:
            started = time.perf_counter()
            correlation_id = str(uuid.uuid4())
            logger.info(
                "master_plan_step_started",
                extra={
                    "event": "master_plan_step_started",
                    "task_id": root_message.task_id,
                    "correlation_id": correlation_id,
                    "step_id": step.step_id,
                    "agent": step.agent,
                },
            )
            TaskReplyWaiter.begin(correlation_id, 1)
            try:
                payload = {
                    **dict(root_message.content or {}),
                    **step.payload,
                    "task_type": step.task_type,
                }
                if upstream_context:
                    payload["_upstream_context"] = upstream_context
                child = MCPMessage(
                    task_id=root_message.task_id,
                    correlation_id=correlation_id,
                    sender=AGENT_MASTER,
                    target=step.agent,
                    priority=root_message.priority,
                    content=payload,
                )
                await mcp_bus.send_msg(child)
                replies = await TaskReplyWaiter.wait(correlation_id, timeout=self.timeout)
                if not replies:
                    return StepResult(
                        step_id=step.step_id,
                        agent=step.agent,
                        task_type=step.task_type,
                        status="FAILED",
                        success=False,
                        error_code=STEP_TIMEOUT,
                        error_msg="Agent step timed out",
                        correlation_id=correlation_id,
                        latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    )
                reply = replies[0]
                success = bool(reply.content.get("success"))
                data = reply.content.get("data") or {}
                return StepResult(
                    step_id=step.step_id,
                    agent=step.agent,
                    task_type=step.task_type,
                    status="SUCCESS" if success else "FAILED",
                    success=success,
                    data=data,
                    error_code=(
                        ""
                        if success
                        else str(
                            reply.content.get("error_code")
                            or data.get("error_code")
                            or AGENT_FAILED
                        )
                    ),
                    error_msg=str(reply.content.get("error_msg") or ""),
                    correlation_id=correlation_id,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return StepResult(
                    step_id=step.step_id,
                    agent=step.agent,
                    task_type=step.task_type,
                    status="FAILED",
                    success=False,
                    error_code=STEP_EXECUTION_ERROR,
                    error_msg=f"Agent step failed: {type(exc).__name__}",
                    correlation_id=correlation_id,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            finally:
                TaskReplyWaiter.discard(correlation_id)
                logger.info(
                    "master_plan_step_finished",
                    extra={
                        "event": "master_plan_step_finished",
                        "task_id": root_message.task_id,
                        "correlation_id": correlation_id,
                        "step_id": step.step_id,
                        "agent": step.agent,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                )
