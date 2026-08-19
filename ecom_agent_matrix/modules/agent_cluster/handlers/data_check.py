"""数据校验 workflow；风控写入仍保留原 Exec Handler。"""
from __future__ import annotations

import time

from pydantic import ValidationError

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.core.tasking import TaskContext, WorkflowResult, ensure_task_context
from ecom_agent_matrix.core.tasking.result import INVALID_REQUEST, PARTIAL_SUCCESS, SKILL_FAILED
from ecom_agent_matrix.modules.parsers.data_check import parse_data_check_request
from ecom_agent_matrix.modules.utils.llm_explain import llm_explain


def _metadata(started: float, **extra) -> dict:
    return {
        "workflow": "data_check",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        **extra,
    }


async def run_data_check_workflow(task: dict | TaskContext) -> WorkflowResult:
    started = time.perf_counter()
    ctx = ensure_task_context(task)
    try:
        request = parse_data_check_request(ctx)
    except (ValidationError, TypeError, ValueError) as exc:
        return WorkflowResult(
            success=False,
            error_code=INVALID_REQUEST,
            error_msg=f"数据校验请求参数不合法：{exc}",
            data={"query_kind": "data_check"},
            metadata=_metadata(started),
        )

    check_result = await exec_skill(
        "data_integrity_check",
        {
            "scope": request.scope,
            "sku": request.sku,
            "order_no": request.order_no,
            "limit": request.limit,
        },
    )
    if not check_result.success:
        return WorkflowResult(
            success=False,
            error_code=SKILL_FAILED,
            error_msg=check_result.error_msg or "数据校验失败",
            data={
                "query_kind": "data_check",
                "scope": request.scope,
                "sku": request.sku or "",
                "order_no": request.order_no or "",
                "integrity": check_result.data or {},
                "sql": {"skipped": True},
            },
            metadata=_metadata(started, skill_error_code=check_result.error_code),
        )

    sql_data: dict = {"skipped": True}
    sql_error_code = ""
    if request.custom_sql or request.run_nl_sql:
        sql_payload: dict = {"params": request.sql_params}
        if request.custom_sql:
            sql_payload["sql"] = request.custom_sql
        else:
            sql_payload["query"] = request.query
        sql_result = await exec_skill("safe_sql_query", sql_payload)
        sql_data = {
            "skipped": False,
            "success": sql_result.success,
            "error_code": sql_result.error_code,
            "error_msg": sql_result.error_msg,
            "data": sql_result.data or {},
        }
        if not sql_result.success:
            sql_error_code = sql_result.error_code or SKILL_FAILED

    integrity = check_result.data or {}
    passed = bool(integrity.get("passed"))
    issue_count = int(integrity.get("issue_count") or 0)
    fallback = (
        f"范围 {request.scope} 数据校验通过，未发现明显问题。"
        if passed
        else f"范围 {request.scope} 发现 {issue_count} 条问题，建议优先处理高影响字段缺失与异常订单。"
    )
    summary, summary_source, summary_error = await llm_explain(
        system_prompt=(
            "你是电商数据质量分析师。根据校验结果写简短中文汇总："
            "问题类型优先级与修复建议。不要编造未出现的表或字段。"
        ),
        user_prompt=(
            f"scope={request.scope}\nsku={request.sku}\norder_no={request.order_no}\n"
            f"integrity={integrity}\nsql={sql_data}"
        ),
        fallback=fallback,
        max_tokens=int(getattr(settings, "AGENT_LLM_EXPLAIN_MAX_TOKENS", 450) or 450),
    )
    partial = bool(sql_error_code)
    error_msg = "" if passed else f"发现 {issue_count} 条数据问题"
    if partial:
        error_msg += f"; sql: {sql_data.get('error_msg') or 'failed'}"
    return WorkflowResult(
        success=True,
        partial_success=partial,
        error_code=PARTIAL_SUCCESS if partial else "",
        error_msg=error_msg,
        data={
            "query_kind": "data_check",
            "scope": request.scope,
            "sku": request.sku or "",
            "order_no": request.order_no or "",
            "integrity": integrity,
            "sql": sql_data,
            "data_ok": passed,
            "summary": summary,
            "summary_source": summary_source,
            "summary_error": summary_error or None,
        },
        metadata=_metadata(
            started,
            skill_error_codes={"safe_sql_query": sql_error_code} if partial else {},
        ),
    )


async def handle_data_check(task: dict | TaskContext) -> tuple[bool, str, dict]:
    return (await run_data_check_workflow(task)).as_legacy_tuple()
