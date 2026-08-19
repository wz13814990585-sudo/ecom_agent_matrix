"""数据查询 handler：完整性校验 + 只读 SQL。由 Query Agent 调用，不是独立 Agent。风控写入见 exec。"""
from __future__ import annotations

import asyncio
import re
import time

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.modules.skills.data_integrity_check import SUPPORTED_SCOPES
from ecom_agent_matrix.modules.skills.sql_tool import nl_to_readonly_sql
from ecom_agent_matrix.modules.utils.competitor_parse import extract_sku
from ecom_agent_matrix.modules.utils.llm_explain import llm_explain

logger = setup_logger("agent.data_check")

_ORDER_PATTERN = re.compile(r"\b(?:ORD[-_][A-Z0-9_-]+|\d{10,20})\b", re.IGNORECASE)
_DB_QUERY_HINT = re.compile(
    r"查询数据库|查库|有多少|统计|有哪些表|跑sql|执行sql|select\s|count\(",
    re.I,
)


def _extract_order_no(payload: dict) -> str:
    for key in ("order_no", "order_id", "order"):
        val = payload.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    for key in ("query", "user_query", "text"):
        m = _ORDER_PATTERN.search(str(payload.get(key) or ""))
        if m:
            return m.group(0)
    return ""


def _extract_scope(payload: dict) -> str:
    raw = str(payload.get("scope") or payload.get("check_type") or "").strip().lower()
    if raw in SUPPORTED_SCOPES:
        return raw
    text = str(payload.get("query") or payload.get("user_query") or "").lower()
    if any(k in text for k in ("订单", "order", "风控")) and not any(
        k in text for k in ("商品", "goods", "sku 主数据", "主数据")
    ):
        return "order"
    if any(k in text for k in ("商品", "goods", "主数据", "库存字段")) and "订单" not in text:
        return "goods"
    return "full"


async def handle_data_check(payload: dict) -> tuple[bool, str, dict]:
    """只读：商品/订单完整性 + 可选 safe_sql_query。不写 risk_record。"""
    started = time.perf_counter()
    sku = extract_sku(payload)
    order_no = _extract_order_no(payload)
    scope = _extract_scope(payload)
    limit = int(payload.get("limit", 50))
    skill_timeout = float(settings.DATA_CHECK_SKILL_TIMEOUT)

    try:
        check_res = await asyncio.wait_for(
            exec_skill(
                "data_integrity_check",
                {
                    "scope": scope,
                    "sku": sku or None,
                    "order_no": order_no or None,
                    "limit": limit,
                },
            ),
            timeout=skill_timeout,
        )
    except asyncio.TimeoutError:
        return (
            False,
            f"data_integrity_check 超时（>{skill_timeout}s）",
            {"query_kind": "data_check", "scope": scope, "sku": sku, "order_no": order_no},
        )

    sql_data: dict = {"skipped": True}
    custom_sql = str(payload.get("sql") or payload.get("custom_sql") or "").strip()
    query_text = str(payload.get("query") or payload.get("user_query") or "")
    run_nl_sql = False
    if not custom_sql and query_text and _DB_QUERY_HINT.search(query_text):
        mapped, _label, map_err = nl_to_readonly_sql(query_text)
        run_nl_sql = bool(mapped) and not map_err
    if custom_sql or run_nl_sql:
        try:
            sql_payload: dict = {
                "params": payload.get("sql_params") or payload.get("params") or [],
            }
            if custom_sql:
                sql_payload["sql"] = custom_sql
            else:
                sql_payload["query"] = query_text
            sql_res = await asyncio.wait_for(
                exec_skill("safe_sql_query", sql_payload),
                timeout=min(15.0, skill_timeout),
            )
            sql_data = {
                "skipped": False,
                "success": sql_res.success,
                "error_msg": sql_res.error_msg,
                "data": sql_res.data or {},
            }
        except asyncio.TimeoutError:
            sql_data = {
                "skipped": False,
                "success": False,
                "error_msg": "safe_sql_query timeout",
            }

    if not check_res.success:
        return (
            False,
            check_res.error_msg or "数据校验失败",
            {
                "query_kind": "data_check",
                "scope": scope,
                "sku": sku,
                "order_no": order_no,
                "integrity": check_res.data or {},
                "sql": sql_data,
            },
        )

    integrity = check_res.data or {}
    passed = bool(integrity.get("passed"))
    issue_count = int(integrity.get("issue_count") or 0)
    if passed:
        summary_fallback = f"范围 {scope} 数据校验通过，未发现明显问题。"
    else:
        summary_fallback = (
            f"范围 {scope} 发现 {issue_count} 条问题，"
            "建议优先处理高影响字段缺失与异常订单。"
        )
    summary, summary_source, summary_error = await llm_explain(
        system_prompt=(
            "你是电商数据质量分析师。根据校验结果写简短中文汇总："
            "问题类型优先级与修复建议。不要编造未出现的表或字段。"
        ),
        user_prompt=(
            f"scope={scope}\nsku={sku}\norder_no={order_no}\n"
            f"integrity={integrity}\nsql={sql_data}"
        ),
        fallback=summary_fallback,
        max_tokens=int(getattr(settings, "AGENT_LLM_EXPLAIN_MAX_TOKENS", 450) or 450),
    )

    elapsed_ms = (time.perf_counter() - started) * 1000
    extra_err = ""
    if not sql_data.get("skipped") and not sql_data.get("success"):
        extra_err = f"; sql: {sql_data.get('error_msg') or 'failed'}"
    return (
        True,
        ("" if passed else f"发现 {issue_count} 条数据问题") + extra_err,
        {
            "query_kind": "data_check",
            "scope": scope,
            "sku": sku,
            "order_no": order_no,
            "integrity": integrity,
            "sql": sql_data,
            "data_ok": passed,
            "summary": summary,
            "summary_source": summary_source,
            "summary_error": summary_error or None,
            "latency_ms": round(elapsed_ms, 2),
        },
    )


async def handle_risk(payload: dict) -> tuple[bool, str, dict]:
    """写操作：订单风控落库。由 Exec Agent 调用。"""
    order_no = _extract_order_no(payload)
    if not order_no:
        return (
            False,
            "触发风控需要订单号 order_no",
            {"exec_kind": "risk", "order_no": "", "risk": {"skipped": True, "reason": "missing order_no"}},
        )
    total_amount = payload.get("total_amount")
    buy_count = payload.get("buy_count") or payload.get("buy_num")
    if total_amount is None or buy_count is None:
        return (
            False,
            "风控需要 total_amount 与 buy_count/buy_num",
            {
                "exec_kind": "risk",
                "order_no": order_no,
                "risk": {"skipped": True, "reason": "风控需要 total_amount 与 buy_count/buy_num"},
            },
        )
    skill_timeout = min(10.0, float(settings.DATA_CHECK_SKILL_TIMEOUT))
    try:
        risk_res = await asyncio.wait_for(
            exec_skill(
                "order_risk_check",
                {
                    "order_no": order_no,
                    "total_amount": total_amount,
                    "buy_count": buy_count,
                },
            ),
            timeout=skill_timeout,
        )
    except asyncio.TimeoutError:
        return False, "order_risk_check timeout", {"exec_kind": "risk", "order_no": order_no}
    return (
        bool(risk_res.success),
        risk_res.error_msg or "",
        {
            "exec_kind": "risk",
            "order_no": order_no,
            "risk": {
                "success": risk_res.success,
                "error_msg": risk_res.error_msg,
                "data": risk_res.data or {},
            },
        },
    )
