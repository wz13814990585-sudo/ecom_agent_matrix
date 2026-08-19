"""预警中心接口（竞品价监控）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ecom_agent_matrix.api.auth import require_api_key
from ecom_agent_matrix.api.dispatch import dispatch_and_wait, dispatch_to_master
from ecom_agent_matrix.api.schemas import ApiResult, CompetitorWarnRequest
from ecom_agent_matrix.config.constants import AGENT_QUERY, MSG_PRIORITY_RISK

router = APIRouter(prefix="/api/v1/warn", tags=["warn"], dependencies=[Depends(require_api_key)])


@router.post("/competitor", response_model=ApiResult)
async def competitor_warn(body: CompetitorWarnRequest) -> ApiResult:
    query = (body.query or "").strip()
    if not query and not (body.sku and body.competitor):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="请提供 query，或同时提供 sku + competitor",
        )

    content: dict = {}
    if query:
        content["query"] = query
    if body.sku:
        content["sku"] = body.sku
        content["target_sku"] = body.sku
    if body.competitor:
        content["competitor"] = body.competitor
    if body.compete_price is not None:
        content["compete_price"] = body.compete_price

    if body.via_master:
        content.setdefault("task_type", "competitor_watch")
        result = await dispatch_to_master(
            content,
            priority=MSG_PRIORITY_RISK,
            timeout=body.timeout,
        )
    else:
        if not content.get("sku") or not content.get("competitor"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="直达预警需显式提供 sku 与 competitor",
            )
        result = await dispatch_and_wait(
            target=AGENT_QUERY,
            content=content,
            priority=MSG_PRIORITY_RISK,
            timeout=body.timeout,
        )
    return ApiResult(**result)
