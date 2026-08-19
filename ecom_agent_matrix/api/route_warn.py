"""预警中心接口（竞品价监控）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ecom_agent_matrix.api.auth import get_current_approval_grant, get_current_security_context
from ecom_agent_matrix.api.dispatch import dispatch_and_wait, dispatch_to_master
from ecom_agent_matrix.api.schemas import ApiResult, CompetitorWarnRequest
from ecom_agent_matrix.config.constants import AGENT_QUERY, MSG_PRIORITY_RISK
from ecom_agent_matrix.core.security import SecurityContext, authorize_task
from ecom_agent_matrix.core.security import ApprovalGrant
from ecom_agent_matrix.core.security.errors import AuthorizationError

router = APIRouter(prefix="/api/v1/warn", tags=["warn"])


@router.post("/competitor", response_model=ApiResult)
async def competitor_warn(
    body: CompetitorWarnRequest,
    security: SecurityContext = Depends(get_current_security_context),
    approval: ApprovalGrant | None = Depends(get_current_approval_grant),
) -> ApiResult:
    try:
        authorize_task(security, "competitor_watch")
    except AuthorizationError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="PERMISSION_DENIED") from None
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
            security=security,
            approval=approval,
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
            security=security,
            approval=approval,
        )
    return ApiResult(**result)
