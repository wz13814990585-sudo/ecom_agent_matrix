"""客服领域 TaskContext → CRMRequest 解析。"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ecom_agent_matrix.config.constants import LANG_LIST
from ecom_agent_matrix.core.tasking import TaskContext

_ORDER_PATTERN = re.compile(r"\b(?:ORD[-_][A-Z0-9_-]+|\d{10,20})\b", re.IGNORECASE)
_CJK = re.compile(r"[\u4e00-\u9fff]")


class CRMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str
    session_id: str
    lang: str
    order_no: str | None = None
    use_rag: bool | None = None
    use_taobao: bool = False
    taobao_method: str | None = None
    taobao_payload: dict[str, Any] = Field(default_factory=dict)
    is_fallback_route: bool = False
    task_id: str = ""
    upstream_context: dict[str, Any] = Field(default_factory=dict)


def parse_crm_request(task: TaskContext) -> CRMRequest:
    params = task.params
    lang = (task.lang or ("zh" if _CJK.search(task.query) else "en")).lower()
    if lang not in LANG_LIST:
        lang = "zh" if _CJK.search(task.query) else "en"
    order_no = task.order_no
    if not order_no:
        match = _ORDER_PATTERN.search(task.query)
        order_no = match.group(0) if match else None
    method = str(params.get("taobao_method") or "").strip() or None
    taobao_payload = params.get("taobao_payload") or {}
    return CRMRequest(
        query=task.query,
        session_id=task.session_id or task.task_id,
        lang=lang,
        order_no=order_no,
        use_rag=params.get("use_rag"),
        use_taobao=bool(params.get("use_taobao")) or bool(method),
        taobao_method=method,
        taobao_payload=taobao_payload,
        is_fallback_route=bool(params.get("_fallback_route")),
        task_id=task.task_id,
        upstream_context=(
            params.get("_upstream_context")
            if isinstance(params.get("_upstream_context"), dict)
            else {}
        ),
    )
