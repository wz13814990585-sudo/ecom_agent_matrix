"""LLM 路由：按 LLM_PROVIDER 选择供应商。业务只调用本模块。"""
from __future__ import annotations

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm.provider import LLMProvider
from ecom_agent_matrix.core.llm.providers.deepseek import DeepSeekProvider
from ecom_agent_matrix.core.llm.providers.openai import OpenAIProvider
from ecom_agent_matrix.core.llm.types import ChatMode, ChatResult, LLMError

# 新增供应商：实现 LLMProvider 后在此注册。
PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    "deepseek": DeepSeekProvider,
    "openai": OpenAIProvider,
}

_instances: dict[str, LLMProvider] = {}


def available_providers() -> tuple[str, ...]:
    return tuple(sorted(PROVIDER_REGISTRY))


def get_llm_provider(name: str | None = None) -> LLMProvider:
    """返回供应商实例（按名称缓存；配置在调用时实时读取）。"""
    key = (name or settings.LLM_PROVIDER or "deepseek").strip().lower()
    cls = PROVIDER_REGISTRY.get(key)
    if cls is None:
        raise LLMError(
            f"未知 LLM_PROVIDER={key!r}，可选: {', '.join(available_providers())}"
        )
    inst = _instances.get(key)
    if inst is None:
        inst = cls()
        _instances[key] = inst
    return inst


def is_llm_configured(name: str | None = None) -> bool:
    """当前（或指定）供应商是否已配置 API Key。"""
    try:
        return get_llm_provider(name).is_configured()
    except LLMError:
        return False


def current_provider_name() -> str:
    return get_llm_provider().name


def resolve_mode(mode: ChatMode | str | None = None) -> ChatMode:
    return get_llm_provider().resolve_mode(mode)


def resolve_model(mode: ChatMode | str | None = None) -> str:
    return get_llm_provider().resolve_model(mode)


async def llm_chat(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.2,
    max_tokens: int = 512,
    mode: ChatMode | str | None = None,
) -> ChatResult:
    """通过当前 LLM_PROVIDER 调用 Chat Completions。"""
    return await get_llm_provider().chat(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        mode=mode,
    )


async def llm_reasoner(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
) -> ChatResult:
    """便捷封装：强制 reasoner 模式（忽略 temperature）。"""
    return await llm_chat(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        mode="reasoner",
    )
