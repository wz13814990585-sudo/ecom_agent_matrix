"""DeepSeek 供应商（OpenAI 兼容协议）。"""
from __future__ import annotations

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm.provider import OpenAICompatProvider


class DeepSeekProvider(OpenAICompatProvider):
    name = "deepseek"

    def key_env_name(self) -> str:
        return "DEEPSEEK_API_KEY"

    def api_key(self) -> str:
        return (settings.DEEPSEEK_API_KEY or "").strip()

    def base_url(self) -> str:
        return (settings.DEEPSEEK_BASE_URL or "https://api.deepseek.com").strip()

    def chat_model(self) -> str:
        chat = (settings.DEEPSEEK_CHAT_MODEL or "").strip()
        if chat:
            return chat
        return (settings.DEEPSEEK_MODEL or "deepseek-chat").strip()

    def reasoner_model(self) -> str:
        return (settings.DEEPSEEK_REASONER_MODEL or "deepseek-reasoner").strip()

    def default_mode(self) -> str:
        return (settings.DEEPSEEK_DEFAULT_MODE or settings.LLM_DEFAULT_MODE or "chat").strip()

    def timeout(self) -> float:
        return float(getattr(settings, "DEEPSEEK_TIMEOUT", None) or settings.LLM_TIMEOUT)

    def max_retries(self) -> int:
        return int(getattr(settings, "DEEPSEEK_MAX_RETRIES", None) or settings.LLM_MAX_RETRIES)

    def retry_base_delay(self) -> float:
        return float(
            getattr(settings, "DEEPSEEK_RETRY_BASE_DELAY", None)
            or settings.LLM_RETRY_BASE_DELAY
        )

    def reasoner_min_tokens(self) -> int:
        return int(getattr(settings, "DEEPSEEK_REASONER_MIN_TOKENS", None) or 1024)
