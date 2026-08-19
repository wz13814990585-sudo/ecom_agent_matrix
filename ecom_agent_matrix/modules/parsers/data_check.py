"""数据校验领域 TaskContext → DataCheckRequest 解析。"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ecom_agent_matrix.core.tasking import TaskContext
from ecom_agent_matrix.core.sql import nl_to_readonly_sql

_ORDER_PATTERN = re.compile(r"\b(?:ORD[-_][A-Z0-9_-]+|\d{10,20})\b", re.IGNORECASE)
_SKU_PATTERN = re.compile(r"\bSKU[-_][A-Z0-9_-]+\b", re.IGNORECASE)
_DB_QUERY_HINT = re.compile(
    r"查询数据库|查库|有多少|统计|有哪些表|跑sql|执行sql|select\s|count\(",
    re.I,
)


class DataCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    scope: Literal["goods", "order", "full"] = "full"
    sku: str | None = None
    order_no: str | None = None
    limit: int = Field(default=50, ge=1, le=500)
    custom_sql: str | None = None
    sql_params: list[Any] | dict[str, Any] = Field(default_factory=list)
    run_nl_sql: bool = False
    query: str = ""


def extract_order_no(task: TaskContext) -> str | None:
    if task.order_no:
        return task.order_no
    for key in ("order_id", "order"):
        value = task.params.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    match = _ORDER_PATTERN.search(task.query)
    return match.group(0) if match else None


def _extract_scope(task: TaskContext) -> str:
    raw = str(task.params.get("scope") or task.params.get("check_type") or "").strip().lower()
    if raw in {"goods", "order", "full"}:
        return raw
    lower = task.query.lower()
    if any(key in lower for key in ("订单", "order", "风控")) and not any(
        key in lower for key in ("商品", "goods", "sku 主数据", "主数据")
    ):
        return "order"
    if any(key in lower for key in ("商品", "goods", "主数据", "库存字段")) and "订单" not in lower:
        return "goods"
    return "full"


def parse_data_check_request(task: TaskContext) -> DataCheckRequest:
    params = task.params
    sku = task.sku
    if not sku:
        match = _SKU_PATTERN.search(task.query)
        sku = match.group(0).upper() if match else None
    custom_sql = str(params.get("custom_sql") or params.get("sql") or "").strip() or None
    run_nl_sql = False
    if not custom_sql and task.query and _DB_QUERY_HINT.search(task.query):
        mapped, _label, error = nl_to_readonly_sql(task.query)
        run_nl_sql = bool(mapped) and not error
    return DataCheckRequest(
        scope=_extract_scope(task),
        sku=sku,
        order_no=extract_order_no(task),
        limit=params.get("limit", 50),
        custom_sql=custom_sql,
        sql_params=params.get("sql_params") or params.get("params") or [],
        run_nl_sql=run_nl_sql,
        query=task.query,
    )
