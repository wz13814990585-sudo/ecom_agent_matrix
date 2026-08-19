"""RecoveryDecision 的 fail-closed 执行应用层。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ecom_agent_matrix.config.constants import AGENT_EXEC, AGENT_QUERY, AGENT_RAG
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.security import authorize_task_types
from ecom_agent_matrix.core.security.errors import AuthorizationError
from ecom_agent_matrix.modules.agent_cluster.master.executor import MasterPlanExecutor
from ecom_agent_matrix.modules.agent_cluster.master.planner import TypedMasterPlanner
from ecom_agent_matrix.modules.agent_cluster.master.policy import (
    MasterPlanValidationError,
    validate_master_plan,
)
from ecom_agent_matrix.modules.agent_cluster.master.schemas import (
    MasterPlan,
    PlanExecutionResult,
    RecoveryDecision,
)
from ecom_agent_matrix.modules.agent_cluster.master.telemetry import MasterLLMTelemetry

_SAFE_RETRY_AGENTS = frozenset({AGENT_QUERY, AGENT_RAG})
_EXEC_MAY_HAVE_RUN = frozenset({"RUNNING", "SUCCESS", "FAILED"})


@dataclass(frozen=True)
class RecoveryApplication:
    plan: MasterPlan
    execution: PlanExecutionResult
    decision: RecoveryDecision
    continue_recovery: bool = False
    execution_changed: bool = False


def _safe_original_query(task_input: dict[str, Any]) -> str:
    query = str(
        task_input.get("query")
        or task_input.get("user_query")
        or task_input.get("text")
        or ""
    )[:1200]
    query = re.sub(
        r"(?i)\b(api[_-]?key|token|password|secret)\b\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        query,
    )
    return re.sub(
        r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+",
        "Bearer [REDACTED]",
        query,
    )


def build_replan_input(
    task_input: dict[str, Any],
    execution: PlanExecutionResult,
) -> dict[str, Any]:
    """只给 Planner 失败元数据，不传 Agent data、原 payload 或内部推理。"""
    return {
        "query": _safe_original_query(task_input),
        "_recovery_context": {
            "failed_steps": [
                {
                    "step_id": result.step_id,
                    "agent": result.agent,
                    "task_type": result.task_type,
                    "error_code": result.error_code,
                }
                for result in execution.step_results.values()
                if result.status == "FAILED"
            ],
            "successful_step_ids": [
                result.step_id
                for result in execution.step_results.values()
                if result.status == "SUCCESS"
            ],
        },
    }


def _exec_may_have_run(execution: PlanExecutionResult) -> bool:
    return any(
        result.agent == AGENT_EXEC and result.status in _EXEC_MAY_HAVE_RUN
        for result in execution.step_results.values()
    )


async def apply_recovery_decision(
    decision: RecoveryDecision,
    *,
    plan: MasterPlan,
    execution: PlanExecutionResult,
    root_message: MCPMessage,
    task_input: dict[str, Any],
    executor: MasterPlanExecutor,
    planner: TypedMasterPlanner,
    telemetry: MasterLLMTelemetry,
) -> RecoveryApplication:
    """Apply one bounded recovery action and return the authoritative new execution."""
    if decision.action in {"finish", "clarify"}:
        return RecoveryApplication(plan, execution, decision)

    if decision.action == "retry_agent":
        target = execution.step_results.get(decision.step_id)
        if (
            target is None
            or target.status != "FAILED"
            or target.agent not in _SAFE_RETRY_AGENTS
        ):
            rejected = RecoveryDecision(
                action="finish",
                step_id=decision.step_id,
                reason_code=(
                    "UNSAFE_RECOVERY_RETRY_REJECTED"
                    if target is not None and target.agent == AGENT_EXEC
                    else "INVALID_RECOVERY_RETRY_TARGET"
                ),
            )
            return RecoveryApplication(plan, execution, rejected)
        resumed = await executor.resume(
            plan,
            root_message,
            execution,
            retry_step_ids={decision.step_id},
        )
        return RecoveryApplication(
            plan,
            resumed,
            decision,
            continue_recovery=not resumed.all_success,
            execution_changed=True,
        )

    if decision.action == "replan":
        if _exec_may_have_run(execution):
            rejected = RecoveryDecision(
                action="finish",
                reason_code="UNSAFE_REPLAN_AFTER_EXEC",
            )
            return RecoveryApplication(plan, execution, rejected)

        replanned = await planner.plan(build_replan_input(task_input, execution), telemetry)
        if replanned.decision != "execute":
            rejected = RecoveryDecision(
                action="clarify",
                reason_code=f"REPLAN_{replanned.reason_code}",
                clarification_question=replanned.clarification_question,
            )
            return RecoveryApplication(plan, execution, rejected)
        try:
            validate_master_plan(replanned)
            if root_message.security is not None:
                authorize_task_types(
                    root_message.security,
                    (step.task_type for step in replanned.steps),
                )
        except AuthorizationError:
            rejected = RecoveryDecision(
                action="finish",
                reason_code="PERMISSION_DENIED",
            )
            return RecoveryApplication(plan, execution, rejected)
        except (MasterPlanValidationError, TypeError, ValueError):
            rejected = RecoveryDecision(
                action="finish",
                reason_code="INVALID_REPLANNED_PLAN",
            )
            return RecoveryApplication(plan, execution, rejected)

        retry_ids = {
            step.step_id
            for step in replanned.steps
            if not (
                (old := execution.step_results.get(step.step_id))
                and old.status == "SUCCESS"
                and old.agent == step.agent
                and old.task_type == step.task_type
            )
        }
        resumed = await executor.resume(
            replanned,
            root_message,
            execution,
            retry_step_ids=retry_ids,
        )
        return RecoveryApplication(
            replanned,
            resumed,
            decision,
            continue_recovery=not resumed.all_success,
            execution_changed=True,
        )

    rejected = RecoveryDecision(action="finish", reason_code="INVALID_RECOVERY_ACTION")
    return RecoveryApplication(plan, execution, rejected)


__all__ = [
    "RecoveryApplication",
    "apply_recovery_decision",
    "build_replan_input",
]
