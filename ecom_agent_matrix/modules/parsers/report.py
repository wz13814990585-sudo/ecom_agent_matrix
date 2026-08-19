"""报表领域 TaskContext → ReportRequest 解析。"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ecom_agent_matrix.core.tasking import TaskContext

_DAYS_PATTERN = re.compile(r"(?:近|最近|last)\s*(\d+)\s*(?:天|日|days?)", re.IGNORECASE)
_SUPPORTED = {"daily_ops", "sales", "stock", "risk", "full"}


class UnsupportedReportType(ValueError):
    def __init__(self, report_type: str):
        self.report_type = report_type
        super().__init__(report_type)


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    report_type: Literal["daily_ops", "sales", "stock", "risk", "full"] = "daily_ops"
    days: int = Field(default=7, ge=1, le=90)
    top_k: int = Field(default=5, ge=1, le=20)
    lang: Literal["en", "zh", "es", "fr"] = "zh"
    query: str = ""


def _report_type(task: TaskContext) -> str:
    raw = str(task.params.get("report_type") or task.params.get("type") or "").strip().lower()
    if raw:
        if raw not in _SUPPORTED:
            raise UnsupportedReportType(raw)
        return raw
    lower = task.query.lower()
    if any(key in lower for key in ("销量", "销售", "gmv", "sales")):
        return "sales"
    if any(key in lower for key in ("库存", "stock", "缺货")):
        return "stock"
    if any(key in lower for key in ("风控", "风险", "risk")):
        return "risk"
    if any(key in lower for key in ("完整", "综合", "全量", "full")):
        return "full"
    return "daily_ops"


def parse_report_request(task: TaskContext) -> ReportRequest:
    days = task.params.get("days")
    if days is None:
        match = _DAYS_PATTERN.search(task.query)
        days = int(match.group(1)) if match else 7
    return ReportRequest(
        report_type=_report_type(task),
        days=days,
        top_k=task.params.get("top_k", 5),
        lang=(task.lang or "zh").lower(),
        query=task.query,
    )
