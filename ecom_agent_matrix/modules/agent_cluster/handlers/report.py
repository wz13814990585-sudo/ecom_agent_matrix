"""运营报表 workflow：typed request → ops_report。"""
from __future__ import annotations

import time

from pydantic import ValidationError

from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.core.tasking import TaskContext, WorkflowResult, ensure_task_context
from ecom_agent_matrix.core.tasking.result import (
    INVALID_REQUEST,
    SKILL_FAILED,
    UNSUPPORTED_REPORT_TYPE,
)
from ecom_agent_matrix.modules.parsers.report import (
    UnsupportedReportType,
    parse_report_request,
)
from ecom_agent_matrix.modules.skills.ops_report import SUPPORTED_REPORT_TYPES


def _metadata(started: float, **extra) -> dict:
    return {
        "workflow": "report",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        **extra,
    }


async def run_report_workflow(task: dict | TaskContext) -> WorkflowResult:
    started = time.perf_counter()
    ctx = ensure_task_context(task)
    try:
        request = parse_report_request(ctx)
    except UnsupportedReportType as exc:
        return WorkflowResult(
            success=False,
            error_code=UNSUPPORTED_REPORT_TYPE,
            error_msg=(
                f"不支持的 report_type：{exc.report_type}，"
                f"可选：{', '.join(sorted(SUPPORTED_REPORT_TYPES))}"
            ),
            data={"exec_kind": "ops_report", "supported_report_types": sorted(SUPPORTED_REPORT_TYPES)},
            metadata=_metadata(started),
        )
    except (ValidationError, TypeError, ValueError) as exc:
        return WorkflowResult(
            success=False,
            error_code=INVALID_REQUEST,
            error_msg=f"报表请求参数不合法：{exc}",
            data={"exec_kind": "ops_report"},
            metadata=_metadata(started),
        )

    report_result = await exec_skill(
        "ops_report",
        {
            "report_type": request.report_type,
            "days": request.days,
            "top_k": request.top_k,
            "lang": request.lang,
        },
    )
    if not report_result.success:
        return WorkflowResult(
            success=False,
            error_code=SKILL_FAILED,
            error_msg=report_result.error_msg or "报表生成失败",
            data={
                "exec_kind": "ops_report",
                "report_type": request.report_type,
                "days": request.days,
                "report": report_result.data or {},
            },
            metadata=_metadata(started, skill_error_code=report_result.error_code),
        )

    data = report_result.data or {}
    return WorkflowResult(
        success=True,
        data={
            "exec_kind": "ops_report",
            "report_type": data.get("report_type", request.report_type),
            "days": data.get("days", request.days),
            "summary": data.get("summary", ""),
            "structured": data.get("structured") or {},
            "sections": data.get("sections", {}),
            "source": data.get("source", ""),
            "lang": data.get("lang", request.lang),
            "llm_error": data.get("llm_error"),
        },
        metadata=_metadata(started),
    )


async def handle_report(task: dict | TaskContext) -> tuple[bool, str, dict]:
    return (await run_report_workflow(task)).as_legacy_tuple()
