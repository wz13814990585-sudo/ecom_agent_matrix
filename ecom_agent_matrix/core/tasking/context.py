"""跨 Agent / Workflow 使用的统一任务上下文。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskContext(BaseModel):
    """只承载通用任务字段，并通过 params 保留完整业务参数。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task_id: str = ""
    correlation_id: str = ""
    source_agent: str = ""

    query: str = ""
    task_type: str | None = None
    query_kind: str | None = None
    exec_kind: str | None = None

    sku: str | None = None
    product_name: str | None = None
    lang: str | None = None

    store_id: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None

    order_no: str | None = None
    campaign_id: str | None = None
    competitor: str | None = None
    platform: str | None = None

    params: dict[str, Any] = Field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """为旧只读调用提供最小 mapping 兼容；新代码应使用 typed 属性。"""
        return self.to_payload()[key]

    def to_payload(self) -> dict[str, Any]:
        """返回兼容旧 Handler 的独立业务 payload，不泄露消息信封字段。"""
        payload = deepcopy(self.params)
        for envelope_field in ("task_id", "correlation_id", "source_agent"):
            payload.pop(envelope_field, None)

        payload["query"] = self.query
        optional_fields = (
            "task_type",
            "query_kind",
            "exec_kind",
            "sku",
            "product_name",
            "lang",
            "store_id",
            "tenant_id",
            "user_id",
            "session_id",
            "order_no",
            "campaign_id",
            "competitor",
            "platform",
        )
        for field_name in optional_fields:
            payload.pop(field_name, None)
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                payload[field_name] = value
        return payload

    def with_updates(self, **updates: Any) -> "TaskContext":
        """基于当前上下文创建新对象，不修改原上下文及其 params。"""
        values = self.model_dump()
        values["params"] = deepcopy(self.params)
        values.update(deepcopy(updates))
        return type(self).model_validate(values)
