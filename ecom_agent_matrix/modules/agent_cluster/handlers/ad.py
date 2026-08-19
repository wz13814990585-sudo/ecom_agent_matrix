"""广告优化 handler：调出价 / 预算。由 Exec Agent 调用，不是独立 Agent。"""
from __future__ import annotations

import asyncio
import re
import time

from ecom_agent_matrix.config.constants import AGENT_EXEC
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.memory.long_vector_memory import AgentLongVectorMemory
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.modules.skills.ad_optimize import SUPPORTED_AD_PLATFORMS
from ecom_agent_matrix.modules.utils.competitor_parse import extract_sku

logger = setup_logger("agent.ad")

_long_mem: AgentLongVectorMemory | None = None

_NUM = re.compile(
    r"(?:spend|消耗|花费)[:：\s]*([0-9]+(?:\.[0-9]+)?)"
    r"|(?:revenue|成交|营收)[:：\s]*([0-9]+(?:\.[0-9]+)?)"
    r"|(?:clicks?|点击)[:：\s]*([0-9]+)"
    r"|(?:conversions?|转化)[:：\s]*([0-9]+)",
    re.IGNORECASE,
)

_PLATFORM_ALIASES = {
    "meta": "meta",
    "facebook": "meta",
    "fb": "meta",
    "instagram ads": "meta",
    "google": "google",
    "google ads": "google",
    "tiktok": "tiktok",
    "tiktok ads": "tiktok",
    "amazon": "amazon",
    "amazon ads": "amazon",
}

def _mem() -> AgentLongVectorMemory:
    global _long_mem
    if _long_mem is None:
        _long_mem = AgentLongVectorMemory()
    return _long_mem


def _extract_platform(payload: dict) -> tuple[str | None, str]:
    raw = ""
    for key in ("platform", "ad_platform", "channel"):
        if payload.get(key):
            raw = str(payload[key]).strip().lower()
            break
    if not raw:
        text = str(payload.get("query") or payload.get("user_query") or "").lower()
        for alias in sorted(_PLATFORM_ALIASES.keys(), key=len, reverse=True):
            if alias in text:
                raw = alias
                break
    if not raw:
        return "meta", ""
    mapped = _PLATFORM_ALIASES.get(raw, raw)
    if mapped not in SUPPORTED_AD_PLATFORMS:
        return None, (
            f"不支持的广告平台：{raw}，可选：{', '.join(sorted(SUPPORTED_AD_PLATFORMS))}"
        )
    return mapped, ""


def _f(payload: dict, *keys: str, default=None):
    for key in keys:
        if key in payload and payload[key] is not None and str(payload[key]).strip() != "":
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                return default
    return default


def _parse_metrics_from_query(text: str) -> dict:
    out: dict = {}
    for m in _NUM.finditer(text or ""):
        if m.group(1) is not None:
            out["spend"] = float(m.group(1))
        if m.group(2) is not None:
            out["revenue"] = float(m.group(2))
        if m.group(3) is not None:
            out["clicks"] = float(m.group(3))
        if m.group(4) is not None:
            out["conversions"] = float(m.group(4))
    return out


def _build_skill_params(payload: dict) -> dict:
    q = str(payload.get("query") or payload.get("user_query") or "")
    from_q = _parse_metrics_from_query(q)
    return {
        "sku": extract_sku(payload),
        "campaign_id": str(payload.get("campaign_id") or "").strip(),
        "platform": payload.get("_platform") or "meta",
        "spend": _f(payload, "spend", "cost", "ad_spend", default=from_q.get("spend", 0)),
        "clicks": _f(payload, "clicks", "click", default=from_q.get("clicks", 0)),
        "conversions": _f(
            payload, "conversions", "orders", "cv", default=from_q.get("conversions", 0)
        ),
        "revenue": _f(payload, "revenue", "gmv", "sales", default=from_q.get("revenue", 0)),
        "daily_budget": _f(payload, "daily_budget", "budget"),
        "bid": _f(payload, "bid", "cpc_bid"),
        "target_roas": _f(payload, "target_roas", "roas_target", default=2.0),
    }


async def handle_ad(payload: dict) -> tuple[bool, str, dict]:
    """广告优化：解析投放指标 → ad_optimize →（可选）利润测算。"""
    started = time.perf_counter()
    skill_timeout = float(settings.AD_SKILL_TIMEOUT)
    long_mem = _mem()
    platform, platform_err = _extract_platform(payload)
    sku = extract_sku(payload)

    if platform_err:
        return (
            False,
            platform_err,
            {
                "exec_kind": "ad_optimize",
                "sku": sku,
                "supported_platforms": sorted(SUPPORTED_AD_PLATFORMS),
            },
        )

    payload = {**payload, "_platform": platform}
    skill_params = _build_skill_params(payload)
    sku = skill_params["sku"] or sku

    has_signal = any(
        float(skill_params.get(k) or 0) > 0
        for k in ("spend", "clicks", "conversions", "revenue")
    )
    if not has_signal and not skill_params.get("campaign_id") and not sku:
        return (
            False,
            "缺少投放数据：请提供 spend/clicks/conversions/revenue，或 sku / campaign_id",
            {"exec_kind": "ad_optimize", "sku": sku, "platform": platform},
        )

    history_hits: list = []
    if sku:
        try:
            history_hits = await long_mem.recall(
                query_text=f"sku:{sku} 广告优化 {platform}",
                agent_name=AGENT_EXEC,
                top_k=3,
                meta_filter={"sku": sku},
            )
        except Exception as exc:
            logger.warning(
                "ad_memory_recall_failed",
                extra={"event": "ad_memory_recall_failed", "error": str(exc)},
            )

    try:
        ad_res = await asyncio.wait_for(
            exec_skill("ad_optimize", skill_params),
            timeout=skill_timeout,
        )
    except asyncio.TimeoutError:
        return (
            False,
            f"ad_optimize 超时（>{skill_timeout}s）",
            {"exec_kind": "ad_optimize", "sku": sku, "platform": platform},
        )

    profit_data: dict = {}
    if all(
        k in payload and payload[k] is not None
        for k in ("cost", "shipping", "commission_rate", "sell_price")
    ):
        try:
            profit_res = await asyncio.wait_for(
                exec_skill(
                    "profit_calc",
                    {
                        "cost": payload["cost"],
                        "shipping": payload["shipping"],
                        "commission_rate": payload["commission_rate"],
                        "sell_price": payload["sell_price"],
                    },
                ),
                timeout=5.0,
            )
            if profit_res.success:
                profit_data = profit_res.data or {}
        except asyncio.TimeoutError:
            profit_data = {"error": "profit_calc timeout"}

    if not ad_res.success:
        return (
            False,
            ad_res.error_msg or "ad_optimize 失败",
            {
                "exec_kind": "ad_optimize",
                "sku": sku,
                "platform": platform,
                "ad_optimize": ad_res.data or {},
                "profit": profit_data,
            },
        )

    plan = (ad_res.data or {}).get("plan") or {}
    action = str(plan.get("action") or "")
    if sku and action and action != "hold":
        await long_mem.safe_save_memory(
            agent_name=AGENT_EXEC,
            content=(
                f"广告优化 sku:{sku} platform:{platform} action:{action} "
                f"bid:{plan.get('bid_adjust_pct')}% "
                f"budget:{plan.get('budget_adjust_pct')}% "
                f"reason:{plan.get('reasoning')}"
            ),
            meta={
                "sku": sku,
                "platform": platform,
                "action": action,
                "bid_adjust_pct": plan.get("bid_adjust_pct"),
                "budget_adjust_pct": plan.get("budget_adjust_pct"),
                "success": True,
                "confidence": 0.8,
                "deprecated": False,
            },
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    return (
        True,
        "",
        {
            "exec_kind": "ad_optimize",
            "sku": sku,
            "platform": platform,
            "campaign_id": skill_params.get("campaign_id"),
            "ad_optimize": ad_res.data,
            "profit": profit_data,
            "history_hits": len(history_hits),
            "history_preview": [
                {"id": h.get("id"), "content": h.get("content"), "meta": h.get("meta")}
                for h in history_hits[:3]
            ],
            "latency_ms": round(elapsed_ms, 2),
        },
    )
