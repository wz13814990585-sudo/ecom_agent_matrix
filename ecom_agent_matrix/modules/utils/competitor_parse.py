"""竞品监控共用解析：店铺名 / 价格（供 Master 与 price_warn_agent）。"""
from __future__ import annotations

import re

# 已知平台别名（可选加速，不是唯一来源）
_KNOWN_ALIASES: dict[str, str] = {
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

# 显式模式：监控 X 上 / on X / 竞品店铺 X / 平台：X
_COMPETITOR_PATTERNS = [
    re.compile(
        r"(?:监控|关注|查看|比价)\s*([A-Za-z0-9][\w.&-]{1,40})\s*(?:上|的|里|店铺|平台)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:on|from|at)\s+([A-Za-z0-9][\w.&-]{1,40})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:竞品|对手|店铺|平台|站点|渠道)[:：\s]+([^\s,，。；;]{2,40})",
        re.IGNORECASE,
    ),
    re.compile(
        r"([A-Za-z][\w.&-]{1,30})\s*(?:上的|上该|上此|平台|店铺)",
        re.IGNORECASE,
    ),
]

# 带标签的价格（优先）
_LABELED_PRICE_PATTERNS = [
    re.compile(
        r"(?:compete[_ ]?price|竞品价|现价|售价|报价|单价|价格)[:：\s]*\$?\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Za-z])(?:price)[:：\s]*\$?\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(?:USD|usd)?",
    ),
]

# 应排除的费用上下文（运费等）
_FEE_CONTEXT = re.compile(
    r"(?:运费|邮费|shipping|postage|freight|税费|tax|fee)[:：\s]*\$?\s*[0-9]+(?:\.[0-9]+)?",
    re.IGNORECASE,
)

_STOP_COMPETITOR_TOKENS = frozenset(
    {
        "sku",
        "the",
        "this",
        "that",
        "price",
        "monitor",
        "watch",
        "check",
        "query",
        "商品",
        "价格",
        "监控",
        "预警",
        "库存",
    }
)


def extract_sku(payload: dict) -> str:
    for key in ("target_sku", "sku", "product_sku", "goods_sku", "best_sku"):
        val = payload.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()

    candidates = (
        payload.get("_goods_candidates")
        or payload.get("goods_candidates")
        or payload.get("candidates")
        or []
    )
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, dict) and first.get("sku"):
            return str(first["sku"]).strip()

    for key in ("query", "user_query", "text"):
        matched = _SKU_PATTERN.search(str(payload.get(key) or ""))
        if matched:
            return matched.group(0).upper()
    return ""


def _normalize_competitor_name(raw: str) -> str:
    name = re.sub(r"\s+", " ", str(raw or "").strip(" \t\n\r\"'，,。.;；:："))
    if not name:
        return ""
    key = name.lower()
    if key in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[key]
    # 去除尾部无意义词
    name = re.sub(r"(店铺|平台|站点|商城)$", "", name).strip()
    if key in _STOP_COMPETITOR_TOKENS or name.lower() in _STOP_COMPETITOR_TOKENS:
        return ""
    if _SKU_PATTERN.fullmatch(name):
        return ""
    # 纯数字不像店铺名
    if re.fullmatch(r"[0-9.]+", name):
        return ""
    return name


def extract_competitor(payload: dict) -> str:
    """
    竞品店铺识别：显式字段 > 句式抽取 > 已知别名命中。
    不再依赖硬编码列表作为唯一途径。
    """
    for key in ("competitor", "competitor_name", "shop", "platform"):
        val = payload.get(key)
        if val is not None and str(val).strip():
            return _normalize_competitor_name(str(val))

    text = str(payload.get("query") or payload.get("user_query") or payload.get("text") or "")
    if not text.strip():
        return ""

    for pat in _COMPETITOR_PATTERNS:
        m = pat.search(text)
        if m:
            name = _normalize_competitor_name(m.group(1))
            if name:
                return name

    # 已知别名兜底（大小写不敏感子串）
    lower = text.lower()
    # 长名优先，避免 ali 误伤
    for alias in sorted(_KNOWN_ALIASES.keys(), key=len, reverse=True):
        if alias in lower:
            return _KNOWN_ALIASES[alias]
    return ""


def extract_compete_price(payload: dict) -> float | None:
    """
    提取竞品现价：优先结构化字段与带标签价格；
    屏蔽「运费/shipping」等费用数字，降低「价格29.99，运费5」误匹配风险。
    """
    for key in ("compete_price", "competitor_price"):
        if key in payload and payload[key] is not None and str(payload[key]).strip() != "":
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                return None

    # 裸 price 字段仅当不是明显运费上下文时使用
    if "price" in payload and payload["price"] is not None and str(payload["price"]).strip() != "":
        try:
            return float(payload["price"])
        except (TypeError, ValueError):
            return None

    text = str(payload.get("query") or payload.get("user_query") or payload.get("text") or "")
    if not text.strip():
        return None

    # 去掉运费等片段后再匹配，避免抢到错误数字
    cleaned = _FEE_CONTEXT.sub(" ", text)

    for pat in _LABELED_PRICE_PATTERNS:
        m = pat.search(cleaned)
        if m:
            try:
                return float(m.group(1))
            except (TypeError, ValueError):
                continue

    # 最后：商品价语境下的「xx 元」，仍避开运费句
    yuan = re.search(
        r"(?<![运邮船])(?:价|售)[^0-9]{0,6}([0-9]+(?:\.[0-9]+)?)\s*元",
        cleaned,
    )
    if yuan:
        try:
            return float(yuan.group(1))
        except (TypeError, ValueError):
            return None
    return None
