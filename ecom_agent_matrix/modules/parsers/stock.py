"""库存领域 TaskContext → StockRequest 解析。"""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from ecom_agent_matrix.core.tasking import TaskContext

_SKU_PATTERN = re.compile(r"\bSKU[-_][A-Z0-9_-]+\b", re.IGNORECASE)


class StockRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sku: str | None = None
    predict_days: int = Field(default=7, ge=1, le=90)


def parse_stock_request(task: TaskContext) -> StockRequest:
    """只从 canonical sku 或 query 中明确的 SKU 格式读取标识。"""
    sku = task.sku
    if not sku:
        match = _SKU_PATTERN.search(task.query)
        sku = match.group(0).upper() if match else None
    return StockRequest(
        sku=sku,
        predict_days=task.params.get("predict_days", 7),
    )
