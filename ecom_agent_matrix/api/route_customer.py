"""多语种客服 / 知识问答接口（经 Master 规划：RAG 或数据查询）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ecom_agent_matrix.api.auth import get_current_approval_grant, get_current_security_context
from ecom_agent_matrix.api.dispatch import dispatch_to_master
from ecom_agent_matrix.api.schemas import ApiResult, CustomerChatRequest
from ecom_agent_matrix.config.constants import MSG_PRIORITY_CUSTOMER
from ecom_agent_matrix.core.security import SecurityContext, authorize_task
from ecom_agent_matrix.core.security import ApprovalGrant
from ecom_agent_matrix.core.security.errors import AuthorizationError
from fastapi import HTTPException, status

router = APIRouter(
    prefix="/api/v1/customer",
    tags=["customer"],
)


@router.post("/chat", response_model=ApiResult)
async def customer_chat(
    body: CustomerChatRequest,
    security: SecurityContext = Depends(get_current_security_context),
    approval: ApprovalGrant | None = Depends(get_current_approval_grant),
) -> ApiResult:
    content: dict = {
        "query": body.query,
        "user_query": body.query,
        "lang": body.lang,
        "use_taobao": body.use_taobao,
        "task_type": "order_query" if (body.use_taobao or body.order_no) else "knowledge_qa",
    }
    if body.session_id:
        content["session_id"] = body.session_id
    if body.use_rag is not None:
        content["use_rag"] = body.use_rag
    if body.order_no:
        content["order_no"] = body.order_no
    if body.taobao_method:
        content["taobao_method"] = body.taobao_method

    try:
        authorize_task(security, str(content["task_type"]))
    except AuthorizationError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="PERMISSION_DENIED") from None

    result = await dispatch_to_master(
        content,
        priority=MSG_PRIORITY_CUSTOMER,
        timeout=body.timeout,
        security=security,
        approval=approval,
    )
    return ApiResult(**result)
