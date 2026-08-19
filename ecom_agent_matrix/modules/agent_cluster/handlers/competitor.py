"""竞品价格 workflow：单平台监控或多平台并发比价。"""
from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError

from ecom_agent_matrix.config.constants import AGENT_QUERY
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.memory.long_vector_memory import AgentLongVectorMemory
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.core.tasking import TaskContext, WorkflowResult, ensure_task_context
from ecom_agent_matrix.core.tasking.result import (
    INVALID_REQUEST,
    MISSING_COMPETITOR,
    MISSING_SKU,
    PARTIAL_SUCCESS,
    PRICE_UNAVAILABLE,
    SKILL_FAILED,
    WORKFLOW_TIMEOUT,
)
from ecom_agent_matrix.modules.parsers.competitor import (
    CompetitorRequest,
    parse_competitor_request,
)
from ecom_agent_matrix.modules.utils.llm_explain import llm_explain

_long_mem: AgentLongVectorMemory | None = None


def _mem() -> AgentLongVectorMemory:
    global _long_mem
    if _long_mem is None:
        _long_mem = AgentLongVectorMemory()
    return _long_mem


def _metadata(started: float, **extra) -> dict:
    return {
        "workflow": "competitor",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        **extra,
    }


async def _run_multi_compare(
    request: CompetitorRequest,
    started: float,
) -> WorkflowResult:
    async def fetch(platform: str):
        return await exec_skill(
            "competitor_price",
            {
                "target_sku": request.sku,
                "sku": request.sku,
                "competitor": platform,
                "query": request.query,
            },
        )

    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(fetch(platform) for platform in request.platforms),
                return_exceptions=True,
            ),
            timeout=float(settings.QUERY_SKILL_TIMEOUT),
        )
    except asyncio.TimeoutError:
        return WorkflowResult(
            success=False,
            error_code=WORKFLOW_TIMEOUT,
            error_msg=f"多平台比价 workflow 超时（>{settings.QUERY_SKILL_TIMEOUT}s）",
            data={
                "query_kind": "competitor",
                "mode": "multi_compare",
                "target_sku": request.sku,
                "platforms": request.platforms,
                "comparisons": [],
            },
            metadata=_metadata(started),
        )

    comparisons: list[dict] = []
    skill_error_codes: dict[str, str] = {}
    for platform, result in zip(request.platforms, results):
        if isinstance(result, BaseException):
            comparisons.append(
                {"competitor": platform, "compete_price": None, "error": type(result).__name__}
            )
            skill_error_codes[platform] = SKILL_FAILED
            continue
        if result.success and result.data.get("compete_price") is not None:
            comparisons.append(
                {
                    "competitor": platform,
                    "compete_price": float(result.data["compete_price"]),
                    "price_source": result.data.get("price_source"),
                    "source_ref": result.data.get("source_ref"),
                    "currency": result.data.get("currency") or "USD",
                }
            )
        else:
            comparisons.append(
                {
                    "competitor": platform,
                    "compete_price": None,
                    "error": result.error_msg or "无报价",
                }
            )
            skill_error_codes[platform] = result.error_code or SKILL_FAILED

    priced = [row for row in comparisons if row.get("compete_price") is not None]
    lowest = min(priced, key=lambda row: row["compete_price"]) if priced else None
    highest = max(priced, key=lambda row: row["compete_price"]) if priced else None
    summary = f"SKU {request.sku} 多平台比价：" + (
        "；".join(f"{row['competitor']}={row['compete_price']}" for row in priced)
        if priced
        else "暂无有效报价"
    )
    if lowest:
        summary += f"。最低价 {lowest['competitor']} {lowest['compete_price']}"

    partial = bool(priced) and len(priced) < len(request.platforms)
    success = bool(priced)
    error_code = PARTIAL_SUCCESS if partial else ("" if success else SKILL_FAILED)
    advice = summary
    advice_source = "template"
    advice_error = ""
    if success:
        advice, advice_source, advice_error = await llm_explain(
            system_prompt=(
                "你是跨境电商定价顾问。根据多平台比价结果写简短中文解读："
                "谁更低、价差含义、是否建议跟价。不要编造未提供的成本。"
            ),
            user_prompt=str(comparisons),
            fallback=summary,
            max_tokens=int(getattr(settings, "AGENT_LLM_EXPLAIN_MAX_TOKENS", 450) or 450),
        )
    return WorkflowResult(
        success=success,
        partial_success=partial,
        error_code=error_code,
        error_msg="" if success and not partial else (
            "; ".join(f"{row['competitor']}: {row.get('error')}" for row in comparisons if row.get("error"))
            or "所有平台均无有效报价"
        ),
        data={
            "query_kind": "competitor",
            "mode": "multi_compare",
            "target_sku": request.sku,
            "platforms": request.platforms,
            "comparisons": comparisons,
            "lowest": lowest,
            "highest": highest,
            "summary": summary,
            "is_trigger_warn": False,
            "warn_message": "",
            "advice": advice,
            "advice_source": advice_source,
            "advice_error": advice_error or None,
        },
        metadata=_metadata(started, skill_error_codes=skill_error_codes),
    )


async def run_competitor_workflow(task: dict | TaskContext) -> WorkflowResult:
    started = time.perf_counter()
    ctx = ensure_task_context(task)
    try:
        request = parse_competitor_request(ctx)
    except (ValidationError, TypeError, ValueError) as exc:
        return WorkflowResult(
            success=False,
            error_code=INVALID_REQUEST,
            error_msg=f"竞品请求参数不合法：{exc}",
            data={"query_kind": "competitor"},
            metadata=_metadata(started),
        )

    if not request.sku:
        return WorkflowResult(
            success=False,
            error_code=MISSING_SKU,
            error_msg="缺少 target_sku，请提供 SKU 或先完成商品检索",
            data={"query_kind": "competitor", "target_sku": ""},
            metadata=_metadata(started),
        )
    if request.mode == "multi":
        return await _run_multi_compare(request, started)
    if not request.competitor:
        return WorkflowResult(
            success=False,
            error_code=MISSING_COMPETITOR,
            error_msg="缺少 competitor，请指定平台或启用多平台比价",
            data={"query_kind": "competitor", "target_sku": request.sku},
            metadata=_metadata(started),
        )

    compete_price = request.compete_price
    price_meta: dict = {}
    if compete_price is None:
        price_result = await exec_skill(
            "competitor_price",
            {
                "target_sku": request.sku,
                "sku": request.sku,
                "competitor": request.competitor,
                "query": request.query,
            },
        )
        if not price_result.success or price_result.data.get("compete_price") is None:
            return WorkflowResult(
                success=False,
                error_code=PRICE_UNAVAILABLE,
                error_msg=price_result.error_msg or "竞品现价获取失败",
                data={
                    "query_kind": "competitor",
                    "target_sku": request.sku,
                    "competitor": request.competitor,
                    "compete_price": None,
                    "need_clarification": True,
                },
                metadata=_metadata(started, skill_error_code=price_result.error_code),
            )
        compete_price = float(price_result.data["compete_price"])
        price_meta = {
            "price_source": price_result.data.get("price_source"),
            "source_ref": price_result.data.get("source_ref"),
            "price_fallback": True,
        }

    history_hits: list = []
    try:
        history_hits = await _mem().recall(
            query_text=f"sku:{request.sku} 竞品告警 {request.competitor}",
            agent_name=AGENT_QUERY,
            top_k=3,
            meta_filter={"sku": request.sku},
            context=ctx,
        )
    except Exception:
        history_hits = []

    monitor_result = await exec_skill(
        "price_monitor",
        {
            "target_sku": request.sku,
            "competitor": request.competitor,
            "compete_price": compete_price,
            "warn_threshold": request.warn_threshold,
        },
    )
    if not monitor_result.success:
        return WorkflowResult(
            success=False,
            error_code=SKILL_FAILED,
            error_msg=monitor_result.error_msg or "price_monitor 执行失败",
            data={
                "query_kind": "competitor",
                "target_sku": request.sku,
                "competitor": request.competitor,
                "compete_price": compete_price,
                "monitor_data": monitor_result.data or {},
                "is_trigger_warn": False,
                "warn_message": "",
                "history_hits": len(history_hits),
            },
            metadata=_metadata(started, skill_error_code=monitor_result.error_code),
        )

    data = monitor_result.data or {}
    is_warn = bool(data.get("is_trigger_warn"))
    warning = str(data.get("warn_message") or "")
    current_offset = float(data.get("current_price_offset", 0) or 0)
    threshold = data.get("warn_threshold", request.warn_threshold)
    fallback = (
        f"竞品 {request.competitor} 对 {request.sku} 报价 {compete_price}，"
        f"偏移 {current_offset}{'已' if is_warn else '未'}触发阈值 {threshold}。"
        "建议核对自家利润后决定是否跟价或观望。"
    )
    advice, advice_source, advice_error = await llm_explain(
        system_prompt=(
            "你是跨境电商定价顾问。根据给定监控结果写简短中文解读。"
            "不要改写数值，不要编造未提供的成本或库存。"
        ),
        user_prompt=str(data),
        fallback=fallback,
        max_tokens=int(getattr(settings, "AGENT_LLM_EXPLAIN_MAX_TOKENS", 450) or 450),
    )
    return WorkflowResult(
        success=True,
        data={
            "query_kind": "competitor",
            "target_sku": request.sku,
            "competitor": request.competitor,
            "compete_price": compete_price,
            "warn_threshold": threshold,
            "monitor_data": data,
            "is_trigger_warn": is_warn,
            "warn_message": warning,
            "advice": advice,
            "advice_source": advice_source,
            "advice_error": advice_error or None,
            "price_meta": price_meta,
            "history_hits": len(history_hits),
            "history_preview": [
                {
                    "id": hit.get("id"),
                    "content": hit.get("content"),
                    "meta": hit.get("meta"),
                    "distance": hit.get("distance"),
                }
                for hit in history_hits[:3]
            ],
        },
        metadata=_metadata(started),
    )


async def handle_price_warn(task: dict | TaskContext) -> tuple[bool, str, dict]:
    return (await run_competitor_workflow(task)).as_legacy_tuple()
