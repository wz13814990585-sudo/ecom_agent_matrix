"""竞品价格查询 handler：只读询价 / 比价 / 监控判定。由 Query Agent 调用，不是独立 Agent。"""
from __future__ import annotations

import re
import time

from ecom_agent_matrix.config.constants import AGENT_QUERY
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.memory.long_vector_memory import AgentLongVectorMemory
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.modules.utils.competitor_parse import (
    extract_compete_price,
    extract_competitor,
    extract_sku,
)
from ecom_agent_matrix.modules.utils.llm_explain import llm_explain

logger = setup_logger("agent.price_warn")

_long_mem: AgentLongVectorMemory | None = None


def _mem() -> AgentLongVectorMemory:
    global _long_mem
    if _long_mem is None:
        _long_mem = AgentLongVectorMemory()
    return _long_mem

# 未指定平台时的多平台比价默认列表
_DEFAULT_COMPARE_PLATFORMS = ("Temu", "Amazon", "AliExpress", "Shein", "Walmart")


def _want_multi_compare(payload: dict, competitor: str) -> bool:
    if payload.get("multi_compare") or payload.get("compare_all"):
        return True
    if competitor:
        return False
    text = str(payload.get("query") or payload.get("user_query") or "")
    return bool(
        re.search(r"比价|竞价对比|价格对比|各平台|多平台|对比一下", text)
        or re.search(r"compar", text, re.I)
    )


async def _multi_platform_compare(sku: str, query: str, platforms: tuple[str, ...] | list[str]) -> dict:
    """对多个平台询价并汇总（不强制触发单平台告警）。"""
    rows: list[dict] = []
    for platform in platforms:
        price_res = await exec_skill(
            "competitor_price",
            {
                "target_sku": sku,
                "sku": sku,
                "competitor": platform,
                "query": query,
            },
        )
        if price_res.success and price_res.data.get("compete_price") is not None:
            rows.append(
                {
                    "competitor": platform,
                    "compete_price": float(price_res.data["compete_price"]),
                    "price_source": price_res.data.get("price_source"),
                    "source_ref": price_res.data.get("source_ref"),
                    "currency": price_res.data.get("currency") or "USD",
                }
            )
        else:
            rows.append(
                {
                    "competitor": platform,
                    "compete_price": None,
                    "error": price_res.error_msg or "无报价",
                }
            )
    priced = [r for r in rows if r.get("compete_price") is not None]
    lowest = min(priced, key=lambda r: r["compete_price"]) if priced else None
    highest = max(priced, key=lambda r: r["compete_price"]) if priced else None
    summary_parts = [f"{r['competitor']}={r['compete_price']}" for r in priced]
    summary = (
        f"SKU {sku} 多平台比价："
        + ("；".join(summary_parts) if summary_parts else "暂无有效报价")
    )
    if lowest:
        summary += f"。最低价 {lowest['competitor']} {lowest['compete_price']}"
    return {
        "mode": "multi_compare",
        "target_sku": sku,
        "platforms": list(platforms),
        "comparisons": rows,
        "lowest": lowest,
        "highest": highest,
        "summary": summary,
        "is_trigger_warn": False,
        "warn_message": "",
    }


async def handle_price_warn(payload: dict) -> tuple[bool, str, dict]:
    """竞品询价 / 多平台比价 / 价格监控（只读查询侧）。"""
    started = time.perf_counter()
    target_sku = extract_sku(payload)
    competitor = extract_competitor(payload)
    compete_price = extract_compete_price(payload)
    warn_threshold = payload.get("warn_threshold", -10)
    query_text = str(payload.get("query") or "")
    long_mem = _mem()

    if target_sku and _want_multi_compare(payload, competitor):
        compare = await _multi_platform_compare(
            target_sku, query_text, _DEFAULT_COMPARE_PLATFORMS
        )
        advice, advice_source, advice_error = await llm_explain(
            system_prompt=(
                "你是跨境电商定价顾问。根据多平台比价结果写简短中文解读："
                "谁更低、价差含义、是否建议跟价。不要编造未提供的成本。"
            ),
            user_prompt=str(compare),
            fallback=compare.get("summary") or "多平台比价完成。",
            max_tokens=int(getattr(settings, "AGENT_LLM_EXPLAIN_MAX_TOKENS", 450) or 450),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return (
            True,
            "",
            {
                "query_kind": "competitor",
                **compare,
                "advice": advice,
                "advice_source": advice_source,
                "advice_error": advice_error or None,
                "latency_ms": round(elapsed_ms, 2),
            },
        )

    missing = []
    if not target_sku:
        missing.append("target_sku")
    if not competitor:
        missing.append("competitor")

    price_meta: dict = {}
    if compete_price is None and target_sku and competitor:
        price_res = await exec_skill(
            "competitor_price",
            {
                "target_sku": target_sku,
                "sku": target_sku,
                "competitor": competitor,
                "query": query_text,
            },
        )
        if price_res.success and price_res.data.get("compete_price") is not None:
            compete_price = float(price_res.data["compete_price"])
            price_meta = {
                "price_source": price_res.data.get("price_source"),
                "source_ref": price_res.data.get("source_ref"),
                "price_fallback": True,
            }
        else:
            missing.append("compete_price")
    elif compete_price is None:
        missing.append("compete_price")

    if missing:
        hints = []
        if "target_sku" in missing:
            hints.append("请提供 SKU 或商品名（如「防水背包」）")
        if "competitor" in missing:
            hints.append("请指定平台（如 Temu），或不指定以启用多平台比价")
        if "compete_price" in missing:
            hints.append("竞品现价获取失败，可稍后重试或手动传入 compete_price")
        return (
            False,
            f"缺少参数：{', '.join(missing)}。" + "；".join(hints),
            {
                "query_kind": "competitor",
                "target_sku": target_sku,
                "competitor": competitor,
                "compete_price": compete_price,
                "is_trigger_warn": False,
                "need_clarification": True,
                "missing": missing,
            },
        )

    history_hits: list = []
    try:
        history_hits = await long_mem.recall(
            query_text=f"sku:{target_sku} 竞品告警 {competitor}",
            agent_name=AGENT_QUERY,
            top_k=3,
            meta_filter={"sku": target_sku},
        )
    except Exception as exc:
        logger.warning(
            "price_warn_memory_recall_failed",
            extra={"event": "price_warn_memory_recall_failed", "error": str(exc)},
        )

    skill_res = await exec_skill(
        "price_monitor",
        {
            "target_sku": target_sku,
            "competitor": competitor,
            "compete_price": compete_price,
            "warn_threshold": warn_threshold,
        },
    )

    if not skill_res.success:
        return (
            False,
            skill_res.error_msg or "price_monitor 执行失败",
            {
                "query_kind": "competitor",
                "target_sku": target_sku,
                "competitor": competitor,
                "compete_price": compete_price,
                "monitor_data": skill_res.data or {},
                "is_trigger_warn": False,
                "warn_message": "",
                "history_hits": len(history_hits),
            },
        )

    data = skill_res.data or {}
    is_warn = bool(data.get("is_trigger_warn"))
    warn_msg = str(data.get("warn_message") or "")
    current_offset = float(data.get("current_price_offset", 0) or 0)
    thr = data.get("warn_threshold", warn_threshold)

    if is_warn:
        await long_mem.safe_save_memory(
            agent_name=AGENT_QUERY,
            content=(
                f"告警 sku:{target_sku} competitor:{competitor} "
                f"price:{compete_price} offset:{current_offset} "
                f"threshold:{thr} msg:{warn_msg}"
            ),
            meta={
                "sku": target_sku,
                "competitor": competitor,
                "compete_price": compete_price,
                "current_price_offset": current_offset,
                "warn_threshold": thr,
                "is_trigger_warn": True,
                "success": True,
                "confidence": 0.85,
                "deprecated": False,
            },
        )

    if is_warn:
        explain_fallback = (
            f"竞品 {competitor} 对 {target_sku} 报价 {compete_price}，"
            f"相对历史最低偏移 {current_offset}（阈值 {thr}）。"
            "建议核对自家利润后决定是否跟价或观望。"
        )
    else:
        explain_fallback = (
            f"竞品 {competitor} 对 {target_sku} 报价 {compete_price}，"
            f"偏移 {current_offset} 未触发阈值 {thr}，可继续观察。"
        )
    advice, advice_source, advice_error = await llm_explain(
        system_prompt=(
            "你是跨境电商定价顾问。根据给定监控结果写简短中文解读："
            "为何告警/未告警、与历史对比含义、建议跟价或观望。"
            "不要改写数值，不要编造未提供的成本或库存。"
        ),
        user_prompt=(
            f"sku={target_sku}\ncompetitor={competitor}\n"
            f"compete_price={compete_price}\n"
            f"current_price_offset={current_offset}\n"
            f"warn_threshold={thr}\n"
            f"is_trigger_warn={is_warn}\n"
            f"warn_message={warn_msg}\n"
            f"monitor_data={data}"
        ),
        fallback=explain_fallback,
        max_tokens=int(getattr(settings, "AGENT_LLM_EXPLAIN_MAX_TOKENS", 450) or 450),
    )

    elapsed_ms = (time.perf_counter() - started) * 1000
    return (
        True,
        "",
        {
            "query_kind": "competitor",
            "target_sku": target_sku,
            "competitor": competitor,
            "compete_price": compete_price,
            "warn_threshold": thr,
            "monitor_data": data,
            "is_trigger_warn": is_warn,
            "warn_message": warn_msg,
            "advice": advice,
            "advice_source": advice_source,
            "advice_error": advice_error or None,
            "price_meta": price_meta,
            "history_hits": len(history_hits),
            "history_preview": [
                {
                    "id": h.get("id"),
                    "content": h.get("content"),
                    "meta": h.get("meta"),
                    "distance": h.get("distance"),
                }
                for h in history_hits[:3]
            ],
            "latency_ms": round(elapsed_ms, 2),
        },
    )
