"""OpenAI 供应商（官方 Chat Completions）。"""
from __future__ import annotations

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm.provider import OpenAICompatProvider
from ecom_agent_matrix.core.llm.types import ChatMode


class OpenAIProvider(OpenAICompatProvider):
    name = "openai"

    def key_env_name(self) -> str:
        return "OPENAI_API_KEY"

    def api_key(self) -> str:
        return (settings.OPENAI_API_KEY or "").strip()

    def base_url(self) -> str:
        return (settings.OPENAI_BASE_URL or "https://api.openai.com/v1").strip()

    def chat_model(self) -> str:
        return (settings.OPENAI_CHAT_MODEL or "gpt-4o-mini").strip()

    def reasoner_model(self) -> str:
        return (settings.OPENAI_REASONER_MODEL or "o4-mini").strip()

    def default_mode(self) -> str:
        return (settings.LLM_DEFAULT_MODE or "chat").strip()

    def timeout(self) -> float:
        return float(getattr(settings, "OPENAI_TIMEOUT", None) or settings.LLM_TIMEOUT)

    def max_retries(self) -> int:
        return int(settings.LLM_MAX_RETRIES)

    def retry_base_delay(self) -> float:
        return float(settings.LLM_RETRY_BASE_DELAY)

    def reasoner_min_tokens(self) -> int:
        return int(getattr(settings, "OPENAI_REASONER_MIN_TOKENS", None) or 1024)

    def use_max_completion_tokens(self, mode: ChatMode) -> bool:
        return mode == "reasoner"
