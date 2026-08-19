"""LLM Provider 路由与供应商差异（不打真实 API）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm import (
    LLMError,
    available_providers,
    get_llm_provider,
    is_llm_configured,
)
from ecom_agent_matrix.core.llm.providers.deepseek import DeepSeekProvider
from ecom_agent_matrix.core.llm.providers.openai import OpenAIProvider


def test_registry_has_deepseek_and_openai():
    names = available_providers()
    assert "deepseek" in names and "openai" in names
    assert get_llm_provider("deepseek").name == "deepseek"
    assert get_llm_provider("openai").name == "openai"


def test_unknown_provider_raises():
    try:
        get_llm_provider("not-a-vendor")
        raise AssertionError("应抛出 LLMError")
    except LLMError as exc:
        assert "未知 LLM_PROVIDER" in str(exc)


def test_deepseek_resolve_model():
    p = DeepSeekProvider()
    assert p.resolve_mode("r1") == "reasoner"
    assert "reasoner" in p.resolve_model("reasoner")
    assert p.resolve_model("chat")


def test_openai_reasoner_payload_uses_max_completion_tokens():
    p = OpenAIProvider()
    payload = p.build_payload(
        model="o4-mini",
        mode="reasoner",
        user_prompt="hi",
        system_prompt="sys",
        temperature=0.9,
        max_tokens=1024,
    )
    assert "max_completion_tokens" in payload
    assert "max_tokens" not in payload
    assert "temperature" not in payload

    chat = p.build_payload(
        model="gpt-4o-mini",
        mode="chat",
        user_prompt="hi",
        system_prompt="sys",
        temperature=0.2,
        max_tokens=256,
    )
    assert chat["max_tokens"] == 256
    assert chat["temperature"] == 0.2


def test_is_configured_follows_selected_provider():
    old_provider = settings.LLM_PROVIDER
    old_ds = settings.DEEPSEEK_API_KEY
    old_oa = settings.OPENAI_API_KEY
    try:
        settings.LLM_PROVIDER = "deepseek"
        settings.DEEPSEEK_API_KEY = ""
        settings.OPENAI_API_KEY = "sk-openai-only"
        assert is_llm_configured() is False
        assert is_llm_configured("openai") is True

        settings.LLM_PROVIDER = "openai"
        assert is_llm_configured() is True
        settings.OPENAI_API_KEY = ""
        assert is_llm_configured() is False
    finally:
        settings.LLM_PROVIDER = old_provider
        settings.DEEPSEEK_API_KEY = old_ds
        settings.OPENAI_API_KEY = old_oa


def test_compat_shim_still_imports():
    from ecom_agent_matrix.core.llm.deepseek_client import (
        DeepSeekChatResult,
        deepseek_chat,
        resolve_mode,
    )

    assert DeepSeekChatResult is not None
    assert callable(deepseek_chat)
    assert resolve_mode("thinking") == "reasoner"


if __name__ == "__main__":
    test_registry_has_deepseek_and_openai()
    test_unknown_provider_raises()
    test_deepseek_resolve_model()
    test_openai_reasoner_payload_uses_max_completion_tokens()
    test_is_configured_follows_selected_provider()
    test_compat_shim_still_imports()
    print("✅ LLM Provider 路由测试通过")
