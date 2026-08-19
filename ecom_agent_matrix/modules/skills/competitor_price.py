"""竞品价格查询 Skill：按 sku + competitor 获取 compete_price。

模式：
- demo：库内缓存 / 本店价合成（本地联调）
- http：调用自有「价格适配器」HTTP API（COMPETITOR_PRICE_API_URL）
         仅对接你方已授权的合法数据源服务，返回 JSON。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.parse import quote, urlencode

import aiohttp

from ecom_agent_matrix.config.constants import TABLE_COMPETITOR, TABLE_GOODS
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill
from ecom_agent_matrix.db.base import AsyncPGClient
from ecom_agent_matrix.modules.utils.competitor_parse import extract_sku

# 演示用平台相对本店价的系数
_PLATFORM_FACTOR = {
    "temu": 0.88,
    "amazon": 1.05,
    "aliexpress": 0.92,
    "shein": 0.85,
    "decathlon": 0.98,
    "rei": 1.08,
    "walmart": 0.95,
}


def _norm_competitor(name: str) -> str:
    return str(name or "").strip()


def _platform_key(name: str) -> str:
    return _norm_competitor(name).lower().replace(" ", "")


def _price_mode() -> str:
    """优先 COMPETITOR_PRICE_MODE；兼容旧环境变量 COMPETITOR_SPIDER_MODE。"""
    mode = str(getattr(settings, "COMPETITOR_PRICE_MODE", "") or "").strip()
    if not mode:
        mode = str(getattr(settings, "COMPETITOR_SPIDER_MODE", "demo") or "demo")
    return mode.lower()


async def _latest_db_price(sku: str, competitor: str) -> float | None:
    """优先复用库内该竞品最近一次报价。"""
    sql = f"""
    SELECT compete_price
    FROM {TABLE_COMPETITOR}
    WHERE target_sku = %s
      AND LOWER(competitor_name) = LOWER(%s)
    ORDER BY crawl_time DESC NULLS LAST, id DESC
    LIMIT 1
    """
    rows = await AsyncPGClient.execute_sql(sql, [sku, competitor])
    if rows and rows[0][0] is not None:
        return float(rows[0][0])
    return None


async def _our_goods_price(sku: str) -> float | None:
    sql = f"SELECT price FROM {TABLE_GOODS} WHERE sku = %s LIMIT 1"
    rows = await AsyncPGClient.execute_sql(sql, [sku])
    if rows and rows[0][0] is not None:
        return float(rows[0][0])
    return None


def _demo_synthesize_price(sku: str, competitor: str, base: float | None) -> float:
    factor = _PLATFORM_FACTOR.get(_platform_key(competitor), 0.97)
    seed = int(hashlib.md5(f"{sku}|{competitor}".encode()).hexdigest()[:6], 16)
    jitter = 0.96 + (seed % 9) * 0.01  # 0.96~1.04
    base_price = base if base and base > 0 else 39.9
    return round(base_price * factor * jitter, 2)


def _build_adapter_url(template: str, sku: str, competitor: str) -> str:
    """
    支持：
    - 带占位符：https://price.example/api?sku={sku}&platform={competitor}
    - 无占位符：自动追加 ?sku=&competitor=（或 &）
    """
    tpl = (template or "").strip()
    if not tpl:
        return ""
    if "{sku}" in tpl or "{competitor}" in tpl:
        return (
            tpl.replace("{sku}", quote(sku, safe=""))
            .replace("{competitor}", quote(competitor, safe=""))
        )
    sep = "&" if "?" in tpl else "?"
    return f"{tpl}{sep}{urlencode({'sku': sku, 'competitor': competitor})}"


def _extract_price_from_payload(body: Any) -> tuple[float | None, str, str]:
    """从适配器 JSON 解析价格；返回 (price, currency, source_ref)。"""
    if not isinstance(body, dict):
        return None, "USD", ""

    currency = str(body.get("currency") or "USD")
    source_ref = str(
        body.get("source_ref") or body.get("page_url") or body.get("url") or ""
    )

    candidates: list[Any] = [
        body.get("compete_price"),
        body.get("competitor_price"),
        body.get("price"),
    ]
    data = body.get("data")
    if isinstance(data, dict):
        candidates.extend(
            [
                data.get("compete_price"),
                data.get("competitor_price"),
                data.get("price"),
            ]
        )
        currency = str(data.get("currency") or currency)
        source_ref = str(
            data.get("source_ref")
            or data.get("page_url")
            or data.get("url")
            or source_ref
        )

    for raw in candidates:
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return float(raw), currency, source_ref
        except (TypeError, ValueError):
            continue
    return None, currency, source_ref


async def _fetch_http_adapter(
    sku: str,
    competitor: str,
) -> tuple[float | None, str, str, str]:
    """
    调用自有价格适配器。
    返回 (price|None, currency, source_ref, error_msg)。
    """
    base = str(getattr(settings, "COMPETITOR_PRICE_API_URL", "") or "").strip()
    if not base:
        return None, "USD", "", "未配置 COMPETITOR_PRICE_API_URL"

    url = _build_adapter_url(base, sku, competitor)
    headers = {"Accept": "application/json"}
    api_key = str(getattr(settings, "COMPETITOR_PRICE_API_KEY", "") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = aiohttp.ClientTimeout(
        total=float(getattr(settings, "COMPETITOR_PRICE_API_TIMEOUT", 10.0) or 10.0)
    )
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    return None, "USD", "", f"适配器 HTTP {resp.status}: {text[:200]}"
                try:
                    body = json.loads(text) if text else {}
                except json.JSONDecodeError:
                    return None, "USD", "", f"适配器返回非 JSON: {text[:120]}"
                price, currency, source_ref = _extract_price_from_payload(body)
                if price is None:
                    return None, currency, source_ref, "适配器响应缺少 compete_price/price"
                return price, currency, source_ref or url, ""
    except Exception as exc:
        return None, "USD", "", f"适配器请求失败: {exc}"


async def _resolve_demo_price(sku: str, competitor: str) -> tuple[float, str, str]:
    """返回 (price, source, source_ref)。"""
    source_ref = f"demo://price/{_platform_key(competitor)}/{sku}"
    cached = await _latest_db_price(sku, competitor)
    if cached is not None:
        seed = int(hashlib.md5(f"refresh|{sku}|{competitor}".encode()).hexdigest()[:4], 16)
        return round(cached * (0.98 + (seed % 5) * 0.01), 2), "demo_cached", source_ref
    our_price = await _our_goods_price(sku)
    return _demo_synthesize_price(sku, competitor, our_price), "demo_synthesize", source_ref


@register_skill
class CompetitorPriceTool(BaseSkill):
    read_only = True
    side_effect = False
    risk_level = "medium"
    skill_name = "competitor_price"
    skill_desc = (
        "竞品价格查询：输入 target_sku + competitor，输出 compete_price；"
        "数据来自库内缓存、demo 合成或自有价格适配器 API"
    )

    async def run(self, params: dict) -> SkillResult:
        try:
            sku = extract_sku(params)

            competitor = _norm_competitor(
                params.get("competitor")
                or params.get("competitor_name")
                or params.get("platform")
                or ""
            )

            if not sku:
                return SkillResult(success=False, error_msg="缺少 target_sku / sku")
            if not competitor:
                return SkillResult(success=False, error_msg="缺少 competitor")

            mode = _price_mode()
            currency = "USD"
            source_ref = ""
            source = mode
            compete_price: float | None = None
            http_error = ""

            if mode == "http":
                price, currency, source_ref, http_error = await _fetch_http_adapter(
                    sku, competitor
                )
                if price is not None:
                    compete_price = price
                    source = "http_adapter"
                else:
                    allow_fallback = bool(
                        getattr(settings, "COMPETITOR_HTTP_FALLBACK_DEMO", True)
                    )
                    if allow_fallback:
                        compete_price, source, source_ref = await _resolve_demo_price(
                            sku, competitor
                        )
                        source = f"http_fallback_{source}"
                    else:
                        return SkillResult(
                            success=False,
                            error_msg=http_error or "HTTP 适配器未返回价格",
                            data={
                                "target_sku": sku,
                                "sku": sku,
                                "competitor": competitor,
                                "fetch_mode": mode,
                                "price_source": "http_failed",
                            },
                        )
            else:
                compete_price, source, source_ref = await _resolve_demo_price(
                    sku, competitor
                )

            return SkillResult(
                success=True,
                data={
                    "target_sku": sku,
                    "sku": sku,
                    "competitor": competitor,
                    "compete_price": compete_price,
                    "currency": currency,
                    "source_ref": source_ref,
                    "price_source": source,
                    "fetch_mode": mode,
                    "adapter_error": http_error or None,
                },
            )
        except Exception as exc:
            return SkillResult(success=False, error_msg=f"竞品价格查询异常：{exc}")
