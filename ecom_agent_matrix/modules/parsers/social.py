"""社媒领域 TaskContext → SocialRequest 解析。"""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from ecom_agent_matrix.config.constants import LANG_LIST
from ecom_agent_matrix.core.tasking import TaskContext
from ecom_agent_matrix.modules.skills.social_media import SUPPORTED_PLATFORMS

_PLATFORM_ALIASES = {
    "tiktok": "tiktok",
    "抖音": "tiktok",
    "instagram": "instagram",
    "ins": "instagram",
    "ig": "instagram",
    "facebook": "facebook",
    "fb": "facebook",
    "twitter": "twitter",
    "x": "twitter",
    "youtube": "youtube",
}

_PLATFORM_PATTERN = re.compile(
    r"(?:platform|平台|渠道)[:：\s]*([A-Za-z\u4e00-\u9fff]+)"
    r"|(?:for|on|给|发)\s*(tiktok|instagram|ins|ig|facebook|fb|twitter|youtube|抖音)\b",
    re.IGNORECASE,
)

_INSTRUCTION_HINT = re.compile(
    r"(帮我|请帮|生成|写一?[份段]|做一?[份个]|文案|社媒|caption|copy|"
    r"对比价格|比价|顺便|监控|预警|库存|备货)",
    re.IGNORECASE,
)


class UnsupportedSocialPlatform(ValueError):
    def __init__(self, platform: str):
        self.platform = platform
        super().__init__(platform)


class SocialRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = ""
    product_name: str | None = None
    platform: str
    feature: str = "high quality best seller"
    lang: str = "en"
    scene: str = "e-commerce product shooting"
    style: str = "bright commercial"


def normalize_social_platform(raw: str) -> str | None:
    key = str(raw or "").strip().lower()
    if not key:
        return None
    mapped = _PLATFORM_ALIASES.get(key, key)
    return mapped if mapped in SUPPORTED_PLATFORMS else None


def _extract_platform(task: TaskContext) -> str:
    raw = task.platform or task.params.get("channel") or task.params.get("social_platform")
    explicit = bool(str(raw or "").strip())
    if not explicit:
        match = _PLATFORM_PATTERN.search(task.query)
        if match:
            raw = match.group(1) or match.group(2) or ""
            explicit = True
        else:
            lower = task.query.lower()
            for alias in sorted(_PLATFORM_ALIASES, key=len, reverse=True):
                if alias.isascii() and len(alias) <= 3:
                    found = bool(
                        re.search(
                            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                            lower,
                        )
                    )
                else:
                    found = alias in lower
                if found:
                    raw = alias
                    explicit = True
                    break
    if not explicit:
        return "tiktok"
    platform = normalize_social_platform(str(raw))
    if platform is None:
        raise UnsupportedSocialPlatform(str(raw).strip())
    return platform


def _extract_product_name(task: TaskContext) -> str | None:
    if task.product_name:
        return task.product_name
    for key in ("product", "title"):
        value = task.params.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    text = task.query
    if not text:
        return None
    patterns = (
        r"(?:为|给|关于)\s*[「\"'【\[]([^」\"'】\]]{2,40})[」\"'】\]]\s*(?:生成|写|做)",
        r"(?:生成|写|做)\s*(?:一份|一段)?\s*[「\"']?([^「\"'\n,，。；;]{2,40}?)"
        r"[」\"']?\s*(?:的)?\s*(?:tiktok|instagram|社媒|文案|caption|copy)",
        r"(?:product|商品|产品)[:：\s]+([^\s,，。;；]{2,40})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        name = match.group(1).strip(" \t「」\"'")
        if normalize_social_platform(name):
            continue
        if _INSTRUCTION_HINT.fullmatch(name):
            continue
        if name.lower() in SUPPORTED_PLATFORMS or name in _PLATFORM_ALIASES:
            continue
        return name
    return None


def _extract_feature(task: TaskContext) -> str:
    for key in ("feature", "selling_point", "卖点", "highlight"):
        value = task.params.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    match = re.search(
        r"(?:卖点|feature|亮点)[:：\s]+([^\n,，。;；]{2,80})",
        task.query,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else "high quality best seller"


def parse_social_request(task: TaskContext) -> SocialRequest:
    """确定性解析社媒参数；不调用外部系统或 Skill。"""
    lang = (task.lang or "en").lower()
    if lang not in LANG_LIST:
        lang = "en"
    return SocialRequest(
        query=task.query,
        product_name=_extract_product_name(task),
        platform=_extract_platform(task),
        feature=_extract_feature(task),
        lang=lang,
        scene=str(task.params.get("scene") or "e-commerce product shooting").strip(),
        style=str(task.params.get("style") or "bright commercial").strip(),
    )
