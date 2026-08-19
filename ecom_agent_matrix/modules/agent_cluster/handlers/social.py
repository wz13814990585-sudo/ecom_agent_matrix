"""社媒营销 handler：生成文案。由 Exec Agent 调用，不是独立 Agent。"""
from __future__ import annotations

import asyncio
import re
import time

from ecom_agent_matrix.config.constants import LANG_LIST
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.rate_limit import acquire_slot
from ecom_agent_matrix.modules.skills.social_media import SUPPORTED_PLATFORMS
from ecom_agent_matrix.core.skill.skill_registry import exec_skill

logger = setup_logger("agent.social")

# 与 social_media_gen skill 对齐的别名 → 规范名
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

# 指令噪声：命中且无明确商品字段时，不当作商品名
_INSTRUCTION_HINT = re.compile(
    r"(帮我|请帮|生成|写一?[份段]|做一?[份个]|文案|社媒|caption|copy|"
    r"对比价格|比价|顺便|监控|预警|库存|备货)",
    re.IGNORECASE,
)


def normalize_platform(raw: str) -> str | None:
    """返回规范平台名；无法识别或不在支持列表则 None。"""
    key = str(raw or "").strip().lower()
    if not key:
        return None
    mapped = _PLATFORM_ALIASES.get(key, key)
    if mapped in SUPPORTED_PLATFORMS:
        return mapped
    return None


def extract_platform(payload: dict) -> tuple[str | None, str]:
    """
    提取平台。返回 (platform|None, error)。
    未指定时默认 tiktok；显式传入不支持平台则报错。
    """
    explicit = False
    raw = ""
    for key in ("platform", "channel", "social_platform"):
        if payload.get(key):
            explicit = True
            raw = str(payload[key])
            break

    if not explicit:
        text = str(payload.get("query") or payload.get("user_query") or "")
        m = _PLATFORM_PATTERN.search(text)
        if m:
            raw = m.group(1) or m.group(2) or ""
            explicit = True
        else:
            lower = text.lower()
            for alias in sorted(_PLATFORM_ALIASES.keys(), key=len, reverse=True):
                if alias in lower:
                    raw = alias
                    explicit = True
                    break

    if not raw:
        return "tiktok", ""

    platform = normalize_platform(raw)
    if platform is None:
        return None, (
            f"不支持的平台：{raw}，可选：{', '.join(sorted(SUPPORTED_PLATFORMS))}"
        )
    return platform, ""


def extract_product_name(payload: dict) -> str:
    """
    提取商品名：优先显式字段与结构化句式。
    禁止把整句指令 query 截断当商品名（避免「帮我生成tiktok文案，顺便对比价格」误传）。
    """
    for key in ("product_name", "product", "goods_name", "name", "title"):
        val = payload.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()

    text = str(payload.get("query") or payload.get("user_query") or "").strip()
    if not text:
        return ""

    patterns = [
        # 为「防水背包」生成 tiktok 文案
        r"(?:为|给|关于)\s*[「\"'【\[]([^」\"'】\]]{2,40})[」\"'】\]]\s*(?:生成|写|做)",
        # 生成 防水背包 的社媒文案 / 写防水背包tiktok文案
        r"(?:生成|写|做)\s*(?:一份|一段)?\s*[「\"']?([^「\"'\n,，。；;]{2,40}?)"
        r"[」\"']?\s*(?:的)?\s*(?:tiktok|instagram|社媒|文案|caption|copy)",
        # 商品：防水背包 / product: xxx
        r"(?:product|商品|产品)[:：\s]+([^\s,，。;；]{2,40})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip(" \t「」\"'")
            # 过滤纯平台名 / 纯指令词
            if normalize_platform(name):
                continue
            if _INSTRUCTION_HINT.fullmatch(name):
                continue
            if name.lower() in SUPPORTED_PLATFORMS or name in _PLATFORM_ALIASES:
                continue
            return name

    # 整句像指令且无商品锚点 → 不猜测
    if _INSTRUCTION_HINT.search(text):
        return ""
    return ""


def extract_feature(payload: dict) -> str:
    for key in ("feature", "selling_point", "卖点", "highlight"):
        val = payload.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    text = str(payload.get("query") or payload.get("user_query") or "")
    m = re.search(r"(?:卖点|feature|亮点)[:：\s]+([^\n,，。;；]{2,80})", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return "high quality best seller"


def _extract_lang(payload: dict) -> str:
    lang = str(payload.get("lang") or "en").strip().lower()
    return lang if lang in LANG_LIST else "en"


def enrich_social_payload(payload: dict) -> dict:
    """供 Master ReAct 注入：尽量带上解析后的 product_name / platform。"""
    out = dict(payload)
    if not str(out.get("product_name") or "").strip():
        name = extract_product_name(out)
        if name:
            out["product_name"] = name
    if not str(out.get("platform") or "").strip():
        platform, _ = extract_platform(out)
        if platform:
            out["platform"] = platform
    if not str(out.get("feature") or "").strip():
        out["feature"] = extract_feature(out)
    return out


async def handle_social(payload: dict) -> tuple[bool, str, dict]:
    """社媒营销：解析参数 → 限时并发生成文案/绘图提示词。"""
    started = time.perf_counter()
    skill_timeout = float(settings.SOCIAL_SKILL_TIMEOUT)
    payload = enrich_social_payload(payload)
    product_name = extract_product_name(payload)
    feature = extract_feature(payload)
    platform, platform_err = extract_platform(payload)
    scene = str(payload.get("scene") or "e-commerce product shooting").strip()
    style = str(payload.get("style") or "bright commercial").strip()
    lang = _extract_lang(payload)

    async with acquire_slot(
        "social",
        limit=int(settings.SOCIAL_MAX_CONCURRENT),
        mode=settings.SOCIAL_RATE_LIMIT_MODE,
        ttl_sec=max(skill_timeout + 5.0, 30.0),
    ) as rate_backend:
        if platform_err:
            return (
                False,
                platform_err,
                {
                    "exec_kind": "social",
                    "product_name": product_name,
                    "platform": str(payload.get("platform") or ""),
                    "supported_platforms": sorted(SUPPORTED_PLATFORMS),
                    "rate_limit_backend": rate_backend,
                },
            )
        if not product_name:
            return (
                False,
                "缺少 product_name：请在 query 中写明「为「商品名」生成文案」",
                {
                    "exec_kind": "social",
                    "product_name": "",
                    "platform": platform,
                    "rate_limit_backend": rate_backend,
                },
            )

        skill_errors: dict[str, str] = {}
        copy_data: dict = {}
        prompt_data: dict = {}
        copy_ok = False
        prompt_ok = False
        try:
            copy_res, prompt_res = await asyncio.wait_for(
                asyncio.gather(
                    exec_skill(
                        "social_media_gen",
                        {
                            "product_name": product_name,
                            "feature": feature,
                            "platform": platform,
                            "lang": lang,
                        },
                    ),
                    exec_skill(
                        "ai_prompt_generate",
                        {
                            "product": product_name,
                            "scene": scene,
                            "style": style,
                        },
                    ),
                    return_exceptions=False,
                ),
                timeout=skill_timeout,
            )
            copy_ok = bool(copy_res.success)
            prompt_ok = bool(prompt_res.success)
            if copy_ok:
                copy_data = copy_res.data or {}
            else:
                skill_errors["social_media_gen"] = copy_res.error_msg or "失败"
            if prompt_ok:
                prompt_data = prompt_res.data or {}
            else:
                skill_errors["ai_prompt_generate"] = prompt_res.error_msg or "失败"
        except asyncio.TimeoutError:
            skill_errors["timeout"] = f"skill 调用超时（>{skill_timeout}s）"

        ok = copy_ok or prompt_ok
        elapsed_ms = (time.perf_counter() - started) * 1000
        return (
            ok,
            "; ".join(f"{k}: {v}" for k, v in skill_errors.items()),
            {
                "exec_kind": "social",
                "product_name": product_name,
                "feature": feature,
                "platform": platform,
                "lang": lang,
                "scene": scene,
                "style": style,
                "social_copy": copy_data,
                "ai_image_prompt": prompt_data,
                "skill_status": {
                    "social_media_gen": copy_ok,
                    "ai_prompt_generate": prompt_ok,
                },
                "skill_errors": skill_errors,
                "partial_success": ok and bool(skill_errors),
                "rate_limit_backend": rate_backend,
                "latency_ms": round(elapsed_ms, 2),
            },
        )
