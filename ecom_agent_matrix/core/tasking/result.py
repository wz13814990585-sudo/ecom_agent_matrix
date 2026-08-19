"""Workflow 层统一结果模型。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field

INVALID_REQUEST = "INVALID_REQUEST"
MISSING_PRODUCT = "MISSING_PRODUCT"
UNSUPPORTED_PLATFORM = "UNSUPPORTED_PLATFORM"
SKILL_FAILED = "SKILL_FAILED"
WORKFLOW_TIMEOUT = "WORKFLOW_TIMEOUT"
PARTIAL_SUCCESS = "PARTIAL_SUCCESS"


class WorkflowResult(BaseModel):
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: str = ""
    error_msg: str = ""
    partial_success: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_legacy_tuple(self) -> tuple[bool, str, dict]:
        """兼容现有 Handler 的 ``(ok, error_msg, data)`` 返回值。"""
        return self.success, self.error_msg, deepcopy(self.data)
