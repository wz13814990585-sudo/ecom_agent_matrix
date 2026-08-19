"""Agent 侧 LLM 解读层：只生成说明文案，不参与数值计算 / 阈值判定。"""
from __future__ import annotations

import asyncio

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm import current_provider_name, is_llm_configured, llm_chat


async def llm_explain(
    *,
    system_prompt: str,
    user_prompt: str,
    fallback: str,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> tuple[str, str, str]:
    """
    返回 (text, source, error)。
    source: {provider} | template | disabled
    """
    if not getattr(settings, "AGENT_LLM_EXPLAIN_ENABLED", True):
        return fallback, "disabled", ""
    if not is_llm_configured():
        return fallback, "template", "no_api_key"
    try:
        raw = await llm_chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            mode="chat",
        )
        text = (raw.content or "").strip()
        if not text:
            return fallback, "template", "empty_content"
        return text, raw.provider or current_provider_name(), ""
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return fallback, "template", type(exc).__name__
