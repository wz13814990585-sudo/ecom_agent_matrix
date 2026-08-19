"""竞品领域 TaskContext → CompetitorRequest 解析。"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ecom_agent_matrix.core.tasking import TaskContext

DEFAULT_COMPARE_PLATFORMS = ("Temu", "Amazon", "AliExpress", "Shein", "Walmart")

_KNOWN_ALIASES = {
    "amazon": "Amazon",
    "aliexpress": "AliExpress",
    "ali express": "AliExpress",
    "shein": "Shein",
    "decathlon": "Decathlon",
    "rei": "REI",
    "temu": "Temu",
    "walmart": "Walmart",
    "ebay": "eBay",
    "shopee": "Shopee",
    "lazada": "Lazada",
    "tiktok": "TikTok",
    "target": "Target",
}
_SKU_PATTERN = re.compile(r"\bSKU[-_][A-Z0-9_-]+\b", re.IGNORECASE)
_COMPETITOR_PATTERNS = (
    re.compile(r"(?:监控|关注|查看|比价)\s*([A-Za-z0-9][\w.&-]{1,40})\s*(?:上|的|里|店铺|平台)?", re.I),
    re.compile(r"(?:on|from|at)\s+([A-Za-z0-9][\w.&-]{1,40})\b", re.I),
    re.compile(r"(?:竞品|对手|店铺|平台|站点|渠道)[:：\s]+([^\s,，。；;]{2,40})", re.I),
    re.compile(r"([A-Za-z][\w.&-]{1,30})\s*(?:上的|上该|上此|平台|店铺)", re.I),
)
_PRICE_PATTERNS = (
    re.compile(r"(?:compete[_ ]?price|竞品价|现价|售价|报价|单价|价格)[:：\s]*\$?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"(?<![A-Za-z])price[:：\s]*\$?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(?:USD|usd)?"),
)
_FEE_CONTEXT = re.compile(
    r"(?:运费|邮费|shipping|postage|freight|税费|tax|fee)[:：\s]*\$?\s*[0-9]+(?:\.[0-9]+)?",
    re.I,
)
_MULTI_HINT = re.compile(r"比价|竞价对比|价格对比|各平台|多平台|对比一下|compar", re.I)
_STOP_COMPETITOR_TOKENS = frozenset(
    {"sku", "price", "monitor", "watch", "check", "query", "商品", "价格", "监控", "预警", "库存"}
)


class CompetitorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sku: str | None = None
    mode: Literal["single", "multi"]
    competitor: str | None = None
    compete_price: float | None = Field(default=None, gt=0)
    warn_threshold: float = Field(default=-10, le=0)
    platforms: list[str] = Field(default_factory=lambda: list(DEFAULT_COMPARE_PLATFORMS))
    query: str = ""


def _normalize_competitor(raw: str) -> str | None:
    name = re.sub(r"\s+", " ", str(raw or "").strip(" \t\n\r\"'，,。.;；:："))
    if not name:
        return None
    mapped = _KNOWN_ALIASES.get(name.lower())
    if mapped:
        return mapped
    name = re.sub(r"(店铺|平台|站点|商城)$", "", name).strip()
    if name.lower() in _STOP_COMPETITOR_TOKENS or _SKU_PATTERN.fullmatch(name):
        return None
    if re.fullmatch(r"[0-9.]+", name):
        return None
    return name or None


def _query_sku(query: str) -> str | None:
    match = _SKU_PATTERN.search(query)
    return match.group(0).upper() if match else None


def _query_competitor(query: str) -> str | None:
    for pattern in _COMPETITOR_PATTERNS:
        match = pattern.search(query)
        if match:
            competitor = _normalize_competitor(match.group(1))
            if competitor:
                return competitor
    lower = query.lower()
    for alias in sorted(_KNOWN_ALIASES, key=len, reverse=True):
        if alias in lower:
            return _KNOWN_ALIASES[alias]
    return None


def _query_price(query: str) -> float | None:
    cleaned = _FEE_CONTEXT.sub(" ", query)
    for pattern in _PRICE_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            return float(match.group(1))
    match = re.search(r"(?<![运邮船])(?:价|售)[^0-9]{0,6}([0-9]+(?:\.[0-9]+)?)\s*元", cleaned)
    return float(match.group(1)) if match else None


def parse_competitor_request(task: TaskContext) -> CompetitorRequest:
    params = task.params
    competitor = _normalize_competitor(task.competitor or task.platform or "")
    if not competitor:
        competitor = _query_competitor(task.query)

    raw_price = params.get("compete_price")
    if raw_price is None:
        raw_price = params.get("competitor_price")
    if raw_price is None:
        raw_price = params.get("price")
    compete_price = raw_price if raw_price not in (None, "") else _query_price(task.query)

    explicit_multi = bool(params.get("multi_compare") or params.get("compare_all"))
    mode = "multi" if explicit_multi or (not competitor and _MULTI_HINT.search(task.query)) else "single"
    raw_platforms = params.get("platforms") or DEFAULT_COMPARE_PLATFORMS
    if isinstance(raw_platforms, str):
        raw_platforms = [item.strip() for item in raw_platforms.split(",") if item.strip()]
    platforms = [
        _normalize_competitor(str(item)) or str(item).strip()
        for item in raw_platforms
        if str(item).strip()
    ]
    return CompetitorRequest(
        sku=task.sku or _query_sku(task.query),
        mode=mode,
        competitor=competitor,
        compete_price=compete_price,
        warn_threshold=params.get("warn_threshold", -10),
        platforms=platforms,
        query=task.query,
    )
