"""多语种客服 / 知识问答接口（经 Master 规划：RAG 或数据查询）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ecom_agent_matrix.api.auth import require_api_key
from ecom_agent_matrix.api.dispatch import dispatch_to_master
from ecom_agent_matrix.api.schemas import ApiResult, CustomerChatRequest
from ecom_agent_matrix.config.constants import MSG_PRIORITY_CUSTOMER

router = APIRouter(
    prefix="/api/v1/customer",
    tags=["customer"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/chat", response_model=ApiResult)
async def customer_chat(body: CustomerChatRequest) -> ApiResult:
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

    result = await dispatch_to_master(
        content,
        priority=MSG_PRIORITY_CUSTOMER,
        timeout=body.timeout,
    )
    return ApiResult(**result)
