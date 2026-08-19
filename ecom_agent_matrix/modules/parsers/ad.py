"""广告领域 TaskContext → AdRequest 解析。"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ecom_agent_matrix.core.tasking import TaskContext

_NUM = re.compile(
    r"(?:spend|消耗|花费)[:：\s]*([0-9]+(?:\.[0-9]+)?)"
    r"|(?:revenue|成交|营收)[:：\s]*([0-9]+(?:\.[0-9]+)?)"
    r"|(?:clicks?|点击)[:：\s]*([0-9]+)"
    r"|(?:conversions?|转化)[:：\s]*([0-9]+)",
    re.IGNORECASE,
)
_SKU_PATTERN = re.compile(r"\bSKU[-_][A-Z0-9_-]+\b", re.IGNORECASE)
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


class UnsupportedAdPlatform(ValueError):
    def __init__(self, platform: str):
        self.platform = platform
        super().__init__(platform)


class IncompleteProfitInputs(ValueError):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(", ".join(missing))


class ProfitInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    cost: float = Field(ge=0)
    shipping: float = Field(ge=0)
    commission_rate: float = Field(ge=0, lt=1)
    sell_price: float = Field(ge=0)


class AdRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    query: str = ""
    sku: str | None = None
    campaign_id: str | None = None
    platform: Literal["meta", "google", "tiktok", "amazon"] = "meta"
    spend: float = Field(default=0, ge=0)
    clicks: int = Field(default=0, ge=0)
    conversions: int = Field(default=0, ge=0)
    revenue: float = Field(default=0, ge=0)
    daily_budget: float | None = Field(default=None, gt=0)
    bid: float | None = Field(default=None, gt=0)
    target_roas: float = Field(default=2.0, gt=0)
    profit: ProfitInputs | None = None

    def skill_params(self) -> dict:
        return self.model_dump(exclude={"query", "profit"})


def _query_metrics(query: str) -> dict:
    values: dict = {}
    for match in _NUM.finditer(query):
        if match.group(1) is not None:
            values["spend"] = float(match.group(1))
        if match.group(2) is not None:
            values["revenue"] = float(match.group(2))
        if match.group(3) is not None:
            values["clicks"] = int(match.group(3))
        if match.group(4) is not None:
            values["conversions"] = int(match.group(4))
    return values


def _explicit(params: dict, names: tuple[str, ...], fallback):
    for name in names:
        if name in params and params[name] is not None and str(params[name]).strip() != "":
            return params[name]
    return fallback


def _platform(task: TaskContext) -> str:
    raw = task.platform or task.params.get("ad_platform") or task.params.get("channel")
    if not raw:
        lower = task.query.lower()
        for alias in sorted(_PLATFORM_ALIASES, key=len, reverse=True):
            if alias in lower:
                raw = alias
                break
    if not raw:
        return "meta"
    key = str(raw).strip().lower()
    mapped = _PLATFORM_ALIASES.get(key, key)
    if mapped not in {"meta", "google", "tiktok", "amazon"}:
        raise UnsupportedAdPlatform(str(raw).strip())
    return mapped


def parse_ad_request(task: TaskContext) -> AdRequest:
    params = task.params
    from_query = _query_metrics(task.query)
    profit_fields = ("cost", "shipping", "commission_rate", "sell_price")
    supplied = {
        name for name in profit_fields
        if name in params and params[name] is not None and str(params[name]).strip() != ""
    }
    if supplied and len(supplied) != len(profit_fields):
        raise IncompleteProfitInputs([name for name in profit_fields if name not in supplied])
    profit = ProfitInputs(**{name: params[name] for name in profit_fields}) if supplied else None
    sku = task.sku
    if not sku:
        match = _SKU_PATTERN.search(task.query)
        sku = match.group(0).upper() if match else None
    return AdRequest(
        query=task.query,
        sku=sku,
        campaign_id=task.campaign_id,
        platform=_platform(task),
        spend=_explicit(params, ("spend", "ad_spend"), from_query.get("spend", 0)),
        clicks=_explicit(params, ("clicks", "click"), from_query.get("clicks", 0)),
        conversions=_explicit(params, ("conversions", "orders", "cv"), from_query.get("conversions", 0)),
        revenue=_explicit(params, ("revenue", "gmv", "sales"), from_query.get("revenue", 0)),
        daily_budget=_explicit(params, ("daily_budget", "budget"), None),
        bid=_explicit(params, ("bid", "cpc_bid"), None),
        target_roas=_explicit(params, ("target_roas", "roas_target"), 2.0),
        profit=profit,
    )
