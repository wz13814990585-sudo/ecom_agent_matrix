"""将旧业务 payload 确定性标准化为 TaskContext。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from ecom_agent_matrix.core.tasking.context import TaskContext

_ENVELOPE_FIELDS = ("task_id", "correlation_id", "source_agent")


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _first_string(payload: dict[str, Any], *names: str) -> str | None:
    for name in names:
        cleaned = _clean_string(payload.get(name))
        if cleaned is not None:
            return cleaned
    return None


def _query(payload: dict[str, Any]) -> str:
    for name in ("query", "user_query", "text", "message"):
        cleaned = _clean_string(payload.get(name))
        if cleaned is not None:
            return cleaned
    content = payload.get("content")
    if isinstance(content, str):
        return content.strip()
    return ""


def normalize_task_context(
    payload: dict[str, Any],
    *,
    task_id: str = "",
    correlation_id: str = "",
    source_agent: str = "",
) -> TaskContext:
    """复制并标准化 payload；信封标识只信任显式参数。"""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    raw = deepcopy(payload)
    params = deepcopy(raw)
    for envelope_field in _ENVELOPE_FIELDS:
        params.pop(envelope_field, None)

    return TaskContext(
        task_id=_clean_string(task_id) or "",
        correlation_id=_clean_string(correlation_id) or "",
        source_agent=_clean_string(source_agent) or "",
        query=_query(raw),
        task_type=_first_string(raw, "task_type", "_inferred_task_type"),
        query_kind=_first_string(raw, "query_kind"),
        exec_kind=_first_string(raw, "exec_kind"),
        sku=_first_string(raw, "sku", "target_sku", "best_sku"),
        product_name=_first_string(raw, "product_name", "goods_name", "name"),
        lang=_first_string(raw, "lang", "language"),
        store_id=_first_string(raw, "store_id"),
        tenant_id=_first_string(raw, "tenant_id"),
        user_id=_first_string(raw, "user_id"),
        session_id=_first_string(raw, "session_id"),
        order_no=_first_string(raw, "order_no"),
        campaign_id=_first_string(raw, "campaign_id"),
        competitor=_first_string(raw, "competitor"),
        platform=_first_string(raw, "platform"),
        params=params,
    )


def ensure_task_context(task: dict[str, Any] | TaskContext) -> TaskContext:
    """兼容旧 dict 调用，并让已标准化上下文直接通过。"""
    if isinstance(task, TaskContext):
        return task
    return normalize_task_context(task)
