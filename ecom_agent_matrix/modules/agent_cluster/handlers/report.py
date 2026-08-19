"""运营报表 handler：生成报表。由 Exec Agent 调用，不是独立 Agent。"""
from __future__ import annotations

import asyncio
import re
import time

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.modules.skills.ops_report import SUPPORTED_REPORT_TYPES

logger = setup_logger("agent.report")

_DAYS_PATTERN = re.compile(r"(?:近|最近|last)\s*(\d+)\s*(?:天|日|days?)", re.IGNORECASE)


def _extract_report_type(payload: dict) -> str:
    raw = str(payload.get("report_type") or payload.get("type") or "").strip().lower()
    if raw in SUPPORTED_REPORT_TYPES:
        return raw
    text = str(payload.get("query") or payload.get("user_query") or "").lower()
    if any(k in text for k in ("销量", "销售", "gmv", "sales")):
        return "sales"
    if any(k in text for k in ("库存", "stock", "缺货")):
        return "stock"
    if any(k in text for k in ("风控", "风险", "risk")):
        return "risk"
    if any(k in text for k in ("完整", "综合", "全量", "full")):
        return "full"
    return "daily_ops"


def _extract_days(payload: dict) -> int:
    if payload.get("days") is not None:
        try:
            return int(payload["days"])
        except (TypeError, ValueError):
            pass
    text = str(payload.get("query") or payload.get("user_query") or "")
    m = _DAYS_PATTERN.search(text)
    if m:
        return max(1, min(90, int(m.group(1))))
    return 7


async def handle_report(payload: dict) -> tuple[bool, str, dict]:
    """运营报表：聚合销售/库存/风控/竞品指标。"""
    started = time.perf_counter()
    skill_timeout = float(settings.REPORT_SKILL_TIMEOUT)
    report_type = _extract_report_type(payload)
    days = _extract_days(payload)
    top_k = int(payload.get("top_k", 5))
    lang = str(payload.get("lang") or "zh").strip().lower()

    if report_type not in SUPPORTED_REPORT_TYPES:
        return (
            False,
            f"不支持的 report_type：{report_type}，可选：{', '.join(sorted(SUPPORTED_REPORT_TYPES))}",
            {"exec_kind": "ops_report", "supported_report_types": sorted(SUPPORTED_REPORT_TYPES)},
        )

    try:
        report_res = await asyncio.wait_for(
            exec_skill(
                "ops_report",
                {
                    "report_type": report_type,
                    "days": days,
                    "top_k": top_k,
                    "lang": lang,
                },
            ),
            timeout=skill_timeout,
        )
    except asyncio.TimeoutError:
        return (
            False,
            f"ops_report 超时（>{skill_timeout}s）",
            {"exec_kind": "ops_report", "report_type": report_type, "days": days},
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    if not report_res.success:
        return (
            False,
            report_res.error_msg or "报表生成失败",
            {
                "exec_kind": "ops_report",
                "report_type": report_type,
                "days": days,
                "report": report_res.data or {},
            },
        )

    data = report_res.data or {}
    return (
        True,
        "",
        {
            "exec_kind": "ops_report",
            "report_type": data.get("report_type", report_type),
            "days": data.get("days", days),
            "summary": data.get("summary", ""),
            "structured": data.get("structured") or {},
            "sections": data.get("sections", {}),
            "source": data.get("source", ""),
            "lang": data.get("lang", lang),
            "llm_error": data.get("llm_error"),
            "latency_ms": round(elapsed_ms, 2),
        },
    )
