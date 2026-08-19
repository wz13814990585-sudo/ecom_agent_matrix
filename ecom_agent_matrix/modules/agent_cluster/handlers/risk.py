"""订单风控写 workflow：typed request → 高风险写 Skill。"""
from __future__ import annotations

import time

from pydantic import ValidationError

from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.core.tasking import TaskContext, WorkflowResult, ensure_task_context
from ecom_agent_matrix.core.tasking.result import INVALID_REQUEST, SKILL_FAILED
from ecom_agent_matrix.modules.parsers.risk import parse_risk_request


def _metadata(started: float, **extra) -> dict:
    return {
        "workflow": "risk",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        **extra,
    }


async def run_risk_workflow(task: dict | TaskContext) -> WorkflowResult:
    started = time.perf_counter()
    ctx = ensure_task_context(task)
    try:
        request = parse_risk_request(ctx)
    except (ValidationError, TypeError, ValueError) as exc:
        return WorkflowResult(
            success=False,
            error_code=INVALID_REQUEST,
            error_msg=f"风控请求参数不合法：{exc}",
            data={"exec_kind": "risk"},
            metadata=_metadata(started),
        )

    risk_result = await exec_skill("order_risk_check", request.model_dump())
    if not risk_result.success:
        return WorkflowResult(
            success=False,
            error_code=SKILL_FAILED,
            error_msg=risk_result.error_msg or "order_risk_check 失败",
            data={
                "exec_kind": "risk",
                "order_no": request.order_no,
                "risk": {
                    "success": False,
                    "error_code": risk_result.error_code,
                    "error_msg": risk_result.error_msg,
                    "data": risk_result.data or {},
                },
            },
            metadata=_metadata(started, skill_error_code=risk_result.error_code),
        )
    return WorkflowResult(
        success=True,
        data={
            "exec_kind": "risk",
            "order_no": request.order_no,
            "risk": {
                "success": True,
                "error_code": risk_result.error_code,
                "error_msg": risk_result.error_msg,
                "data": risk_result.data or {},
            },
        },
        metadata=_metadata(started),
    )


async def handle_risk(task: dict | TaskContext) -> tuple[bool, str, dict]:
    return (await run_risk_workflow(task)).as_legacy_tuple()
