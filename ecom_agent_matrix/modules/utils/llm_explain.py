"""Agent 侧 LLM 解读层：只生成说明文案，不参与数值计算 / 阈值判定。"""
from __future__ import annotations

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm.deepseek_client import deepseek_chat


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
    source: deepseek | template | disabled
    """
    if not getattr(settings, "AGENT_LLM_EXPLAIN_ENABLED", True):
        return fallback, "disabled", ""
    if not settings.DEEPSEEK_API_KEY:
        return fallback, "template", "no_api_key"
    try:
        raw = await deepseek_chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            mode="chat",
        )
        text = (raw.content or "").strip()
        if not text:
            return fallback, "template", "empty_content"
        return text, "deepseek", ""
    except Exception as exc:
        return fallback, "template", str(exc)
