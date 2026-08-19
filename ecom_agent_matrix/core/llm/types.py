"""LLM 供应商无关的结果、模式与异常。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ChatMode = Literal["chat", "reasoner"]


@dataclass
class ChatResult:
    """Chat Completions 结构化结果。"""

    content: str
    reasoning_content: str = ""
    model: str = ""
    mode: str = "chat"
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class LLMError(Exception):
    """LLM 调用基类异常。"""

    def __init__(self, message: str, *, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


class LLMAuthError(LLMError):
    """鉴权失败（401 / 403）。"""


class LLMRateLimitError(LLMError):
    """限流（429），可重试。"""


class LLMServerError(LLMError):
    """服务端错误（5xx，含 503），可重试。"""


class LLMResponseError(LLMError):
    """返回体格式异常或业务侧不可解析。"""


def normalize_mode(mode: ChatMode | str | None = None, *, default: str = "chat") -> ChatMode:
    """规范化 chat / reasoner；非法值回退 default（再不行则 chat）。"""
    raw = (mode or default or "chat").strip().lower()
    if raw in {"reasoner", "reason", "r1", "thinking"}:
        return "reasoner"
    if raw in {"chat", "v3", "fast"}:
        return "chat"
    fallback = (default or "chat").strip().lower()
    return "reasoner" if fallback in {"reasoner", "reason", "r1", "thinking"} else "chat"
