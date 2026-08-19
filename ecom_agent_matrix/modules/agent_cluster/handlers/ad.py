"""广告优化 workflow：typed request → 优化建议 → 可选利润测算。"""
from __future__ import annotations

import time

from pydantic import ValidationError
from ecom_agent_matrix.platform.observability.metrics import observed_workflow

from ecom_agent_matrix.config.constants import AGENT_EXEC
from ecom_agent_matrix.core.memory.long_vector_memory import AgentLongVectorMemory
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.core.tasking import TaskContext, WorkflowResult, ensure_task_context
from ecom_agent_matrix.core.tasking.result import (
    INVALID_REQUEST,
    PARTIAL_SUCCESS,
    SKILL_FAILED,
    UNSUPPORTED_PLATFORM,
)
from ecom_agent_matrix.modules.parsers.ad import (
    IncompleteProfitInputs,
    UnsupportedAdPlatform,
    parse_ad_request,
)
from ecom_agent_matrix.modules.skills.ad_optimize import SUPPORTED_AD_PLATFORMS

_long_mem: AgentLongVectorMemory | None = None


def _mem() -> AgentLongVectorMemory:
    global _long_mem
    if _long_mem is None:
        _long_mem = AgentLongVectorMemory()
    return _long_mem


def _metadata(started: float, **extra) -> dict:
    return {
        "workflow": "ad",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        **extra,
    }


@observed_workflow("ad")
async def run_ad_workflow(task: dict | TaskContext) -> WorkflowResult:
    started = time.perf_counter()
    ctx = ensure_task_context(task)
    try:
        request = parse_ad_request(ctx)
    except UnsupportedAdPlatform as exc:
        return WorkflowResult(
            success=False,
            error_code=UNSUPPORTED_PLATFORM,
            error_msg=(
                f"不支持的广告平台：{exc.platform}，"
                f"可选：{', '.join(sorted(SUPPORTED_AD_PLATFORMS))}"
            ),
            data={"exec_kind": "ad_optimize", "supported_platforms": sorted(SUPPORTED_AD_PLATFORMS)},
            metadata=_metadata(started),
        )
    except IncompleteProfitInputs as exc:
        return WorkflowResult(
            success=False,
            error_code=INVALID_REQUEST,
            error_msg=f"利润测算参数不完整，缺少：{', '.join(exc.missing)}",
            data={"exec_kind": "ad_optimize", "missing_profit_fields": exc.missing},
            metadata=_metadata(started),
        )
    except (ValidationError, TypeError, ValueError) as exc:
        return WorkflowResult(
            success=False,
            error_code=INVALID_REQUEST,
            error_msg=f"广告优化请求参数不合法：{exc}",
            data={"exec_kind": "ad_optimize"},
            metadata=_metadata(started),
        )

    has_signal = any(
        value > 0 for value in (request.spend, request.clicks, request.conversions, request.revenue)
    )
    if not has_signal and not request.campaign_id and not request.sku:
        return WorkflowResult(
            success=False,
            error_code=INVALID_REQUEST,
            error_msg="缺少投放数据：请提供 spend/clicks/conversions/revenue，或 sku / campaign_id",
            data={"exec_kind": "ad_optimize", "platform": request.platform},
            metadata=_metadata(started),
        )

    memory_errors: list[str] = []
    history_hits: list = []
    memory = _mem()
    if request.sku:
        try:
            history_hits = await memory.recall(
                query_text=f"sku:{request.sku} 广告优化 {request.platform}",
                agent_name=AGENT_EXEC,
                top_k=3,
                meta_filter={"sku": request.sku},
                context=ctx,
            )
        except Exception as exc:
            memory_errors.append(f"recall:{type(exc).__name__}")

    ad_result = await exec_skill("ad_optimize", request.skill_params())
    if not ad_result.success:
        return WorkflowResult(
            success=False,
            error_code=SKILL_FAILED,
            error_msg=ad_result.error_msg or "ad_optimize 失败",
            data={
                "exec_kind": "ad_optimize",
                "sku": request.sku or "",
                "platform": request.platform,
                "ad_optimize": ad_result.data or {},
                "profit": {},
            },
            metadata=_metadata(
                started,
                skill_error_code=ad_result.error_code,
                memory_errors=memory_errors,
            ),
        )

    profit_data: dict = {}
    skill_error_codes: dict[str, str] = {}
    errors: list[str] = []
    if request.profit is not None:
        profit_result = await exec_skill("profit_calc", request.profit.model_dump())
        if profit_result.success:
            profit_data = profit_result.data or {}
        else:
            errors.append(f"profit_calc: {profit_result.error_msg or 'failed'}")
            skill_error_codes["profit_calc"] = profit_result.error_code or SKILL_FAILED

    plan = (ad_result.data or {}).get("plan") or {}
    action = str(plan.get("action") or "")
    if request.sku and action and action != "hold":
        try:
            memory_id = await memory.safe_save_memory(
                agent_name=AGENT_EXEC,
                content=(
                    f"广告优化 sku:{request.sku} platform:{request.platform} action:{action} "
                    f"bid:{plan.get('bid_adjust_pct')}% budget:{plan.get('budget_adjust_pct')}%"
                ),
                meta={
                    "sku": request.sku,
                    "platform": request.platform,
                    "action": action,
                    "bid_adjust_pct": plan.get("bid_adjust_pct"),
                    "budget_adjust_pct": plan.get("budget_adjust_pct"),
                    "target_roas": request.target_roas,
                    "metrics_snapshot": plan.get("metrics_snapshot") or {},
                    "success": True,
                    "confidence": 0.8,
                    "deprecated": False,
                },
                context=ctx,
            )
            if memory_id is None:
                memory_errors.append("save:unavailable")
        except Exception as exc:
            memory_errors.append(f"save:{type(exc).__name__}")

    partial = bool(errors or memory_errors)
    return WorkflowResult(
        success=True,
        partial_success=partial,
        error_code=PARTIAL_SUCCESS if partial else "",
        error_msg="; ".join(errors + memory_errors),
        data={
            "exec_kind": "ad_optimize",
            "sku": request.sku or "",
            "platform": request.platform,
            "campaign_id": request.campaign_id or "",
            "ad_optimize": ad_result.data or {},
            "profit": profit_data,
            "history_hits": len(history_hits),
            "history_preview": [
                {"id": hit.get("id"), "content": hit.get("content"), "meta": hit.get("meta")}
                for hit in history_hits[:3]
            ],
        },
        metadata=_metadata(
            started,
            skill_error_codes=skill_error_codes,
            memory_errors=memory_errors,
        ),
    )


async def handle_ad(task: dict | TaskContext) -> tuple[bool, str, dict]:
    return (await run_ad_workflow(task)).as_legacy_tuple()
