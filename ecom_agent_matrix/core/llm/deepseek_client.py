"""兼容层：历史代码可继续 import DeepSeek 名称。

新代码请使用 ecom_agent_matrix.core.llm（llm_chat / ChatResult / LLMError）。
本模块固定走 DeepSeek 供应商，不受 LLM_PROVIDER 切换影响。
"""
from __future__ import annotations

from ecom_agent_matrix.core.llm.http import close_http_session, get_http_session
from ecom_agent_matrix.core.llm.providers.deepseek import DeepSeekProvider
from ecom_agent_matrix.core.llm.types import (
    ChatMode as DeepSeekMode,
    ChatResult as DeepSeekChatResult,
    LLMAuthError as DeepSeekAuthError,
    LLMError as DeepSeekError,
    LLMRateLimitError as DeepSeekRateLimitError,
    LLMResponseError as DeepSeekResponseError,
    LLMServerError as DeepSeekServerError,
)

_provider = DeepSeekProvider()


def resolve_mode(mode: DeepSeekMode | str | None = None) -> DeepSeekMode:
    return _provider.resolve_mode(mode)


def resolve_model(mode: DeepSeekMode | str | None = None) -> str:
    return _provider.resolve_model(mode)


async def deepseek_chat(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.2,
    max_tokens: int = 512,
    mode: DeepSeekMode | str | None = None,
) -> DeepSeekChatResult:
    """兼容入口：始终调用 DeepSeek。"""
    return await _provider.chat(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        mode=mode,
    )


async def deepseek_reasoner(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
) -> DeepSeekChatResult:
    return await deepseek_chat(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        mode="reasoner",
    )


__all__ = [
    "DeepSeekAuthError",
    "DeepSeekChatResult",
    "DeepSeekError",
    "DeepSeekMode",
    "DeepSeekRateLimitError",
    "DeepSeekResponseError",
    "DeepSeekServerError",
    "close_http_session",
    "deepseek_chat",
    "deepseek_reasoner",
    "get_http_session",
    "resolve_mode",
    "resolve_model",
]
