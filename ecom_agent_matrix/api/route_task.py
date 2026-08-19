"""自动化运营任务接口（经 Master ReAct）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ecom_agent_matrix.api.auth import require_api_key
from ecom_agent_matrix.api.dispatch import dispatch_to_master
from ecom_agent_matrix.api.schemas import ApiResult, TaskCreateRequest
from ecom_agent_matrix.config.constants import MSG_PRIORITY_NORMAL

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=ApiResult)
async def create_task(body: TaskCreateRequest) -> ApiResult:
    content = {
        "query": body.query,
        **(body.payload or {}),
    }
    if body.task_type:
        content["task_type"] = body.task_type
    priority = body.priority if body.priority is not None else MSG_PRIORITY_NORMAL
    result = await dispatch_to_master(content, priority=priority, timeout=body.timeout)
    return ApiResult(**result)
