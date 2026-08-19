"""商品检索 workflow：typed request 编排目录或 SKU Skill。"""
from __future__ import annotations

import time

from pydantic import ValidationError
from ecom_agent_matrix.platform.observability.metrics import observed_workflow

from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.core.tasking import TaskContext, WorkflowResult, ensure_task_context
from ecom_agent_matrix.core.tasking.result import INVALID_REQUEST, MISSING_PRODUCT, SKILL_FAILED
from ecom_agent_matrix.modules.parsers.goods import parse_goods_request

logger = setup_logger("agent.goods")


def _metadata(started: float, **extra) -> dict:
    return {
        "workflow": "goods",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        **extra,
    }


@observed_workflow("goods")
async def run_goods_workflow(task: dict | TaskContext) -> WorkflowResult:
    """解析商品请求并编排一个商品 Skill。"""
    started = time.perf_counter()
    ctx = ensure_task_context(task)
    try:
        request = parse_goods_request(ctx)
    except (ValidationError, TypeError, ValueError) as exc:
        return WorkflowResult(
            success=False,
            error_code=INVALID_REQUEST,
            error_msg=f"商品请求参数不合法：{exc}",
            data={"query_kind": "goods", "candidates": [], "best_sku": None},
            metadata=_metadata(started),
        )

    product_name = request.product_name or ""
    if request.mode == "catalog":
        catalog_params: dict = {
            "offset": request.offset,
            "category": request.category,
            "order_by": request.order_by,
            "query": product_name,
            "list_all": request.list_all,
            "store_id": request.store_id,
        }
        if request.limit is not None:
            catalog_params["limit"] = request.limit
        skill_result = await exec_skill("goods_catalog", catalog_params)
        data = skill_result.data or {}
        return WorkflowResult(
            success=skill_result.success,
            error_code="" if skill_result.success else SKILL_FAILED,
            error_msg="" if skill_result.success else skill_result.error_msg,
            data={
                "query_kind": "goods_catalog",
                "mode": "catalog",
                "product_name": product_name,
                "scope": data.get("scope"),
                "store_id": data.get("store_id"),
                "store_name": data.get("store_name"),
                "is_demo_store": data.get("is_demo_store"),
                "is_external": data.get("is_external"),
                "total": data.get("total", 0),
                "count": data.get("count", 0),
                "items": data.get("items", []),
                "summary": data.get("summary", ""),
                "limit": data.get("limit"),
                "offset": data.get("offset"),
                "list_all": data.get("list_all"),
                "truncated": data.get("truncated"),
                "category": data.get("category"),
                "candidates": [],
                "best_sku": None,
            },
            metadata=_metadata(
                started,
                skill_error_code=skill_result.error_code if not skill_result.success else "",
            ),
        )

    if not product_name:
        return WorkflowResult(
            success=False,
            error_code=MISSING_PRODUCT,
            error_msg="缺少商品名，请提供 product_name 或 query；查全库请说「有多少商品/列出商品」",
            data={
                "query_kind": "goods_search",
                "mode": "search",
                "product_name": "",
                "candidates": [],
                "best_sku": None,
            },
            metadata=_metadata(started),
        )

    skill_result = await exec_skill(
        "goods_sku_search",
        {"product_name": product_name, "top_k": request.top_k},
    )
    data = skill_result.data or {}
    candidates = data.get("candidates", [])
    success = skill_result.success and bool(candidates)
    error_msg = (
        skill_result.error_msg
        if not skill_result.success
        else ("未找到匹配商品" if not candidates else "")
    )
    return WorkflowResult(
        success=success,
        error_code="" if success else SKILL_FAILED,
        error_msg=error_msg,
        data={
            "query_kind": "goods_search",
            "mode": "search",
            "product_name": product_name,
            "candidates": candidates,
            "best_sku": data.get("best_sku"),
            "count": data.get("count", 0),
            "match_mode": data.get("match_mode", "none"),
            "semantic_fallback_used": data.get("semantic_fallback_used", False),
            "semantic_error": data.get("semantic_error", ""),
        },
        metadata=_metadata(
            started,
            skill_error_code=skill_result.error_code if not skill_result.success else "",
        ),
    )


async def handle_goods(task: dict | TaskContext) -> tuple[bool, str, dict]:
    """兼容旧 Query Agent/Handler tuple 协议。"""
    return (await run_goods_workflow(task)).as_legacy_tuple()
