"""DeepSeek Chat API 异步客户端（OpenAI 兼容协议，不含业务 Prompt）。

支持 chat / reasoner 双模式：
- mode=\"chat\"     → DEEPSEEK_CHAT_MODEL（默认 deepseek-chat）
- mode=\"reasoner\" → DEEPSEEK_REASONER_MODEL（默认 deepseek-reasoner）
- mode=None        → DEEPSEEK_DEFAULT_MODE
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import aiohttp

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger

logger = setup_logger("llm.deepseek")

DeepSeekMode = Literal["chat", "reasoner"]


# ---------------------------------------------------------------------------
# 结果与异常（供 Skill / Agent 区分处理）
# ---------------------------------------------------------------------------


@dataclass
class DeepSeekChatResult:
    """Chat Completions 结构化结果。"""

    content: str
    reasoning_content: str = ""
    model: str = ""
    mode: str = "chat"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class DeepSeekError(Exception):
    """DeepSeek 调用基类异常。"""

    def __init__(self, message: str, *, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


class DeepSeekAuthError(DeepSeekError):
    """鉴权失败（401 / 403）。"""


class DeepSeekRateLimitError(DeepSeekError):
    """限流（429），可重试。"""


class DeepSeekServerError(DeepSeekError):
    """服务端错误（5xx，含 503），可重试。"""


class DeepSeekResponseError(DeepSeekError):
    """返回体格式异常或业务侧不可解析。"""


# ---------------------------------------------------------------------------
# 模式解析
# ---------------------------------------------------------------------------


def resolve_mode(mode: DeepSeekMode | str | None = None) -> DeepSeekMode:
    """规范化模式；非法值回退到默认。"""
    raw = (mode or settings.DEEPSEEK_DEFAULT_MODE or "chat").strip().lower()
    if raw in {"reasoner", "reason", "r1", "thinking"}:
        return "reasoner"
    if raw in {"chat", "v3", "fast"}:
        return "chat"
    logger.warning(
        "deepseek_invalid_mode",
        extra={"event": "deepseek_invalid_mode", "error": f"未知模式 {mode!r}，回退 chat"},
    )
    return "chat"


def resolve_model(mode: DeepSeekMode | str | None = None) -> str:
    """按模式解析实际 model 名。"""
    resolved = resolve_mode(mode)
    if resolved == "reasoner":
        return (settings.DEEPSEEK_REASONER_MODEL or "deepseek-reasoner").strip()
    # chat：优先 CHAT_MODEL，兼容旧 DEEPSEEK_MODEL
    chat = (settings.DEEPSEEK_CHAT_MODEL or "").strip()
    if chat:
        return chat
    return (settings.DEEPSEEK_MODEL or "deepseek-chat").strip()


# ---------------------------------------------------------------------------
# 全局复用 ClientSession
# ---------------------------------------------------------------------------

_session: aiohttp.ClientSession | None = None
_session_lock = asyncio.Lock()


async def get_http_session() -> aiohttp.ClientSession:
    """获取进程内单例 aiohttp session（懒创建）。"""
    global _session
    if _session is not None and not _session.closed:
        return _session
    async with _session_lock:
        if _session is None or _session.closed:
            timeout = aiohttp.ClientTimeout(total=float(settings.DEEPSEEK_TIMEOUT))
            _session = aiohttp.ClientSession(timeout=timeout)
    return _session


async def close_http_session() -> None:
    """进程退出时关闭 session（可选调用）。"""
    global _session
    async with _session_lock:
        if _session is not None and not _session.closed:
            await _session.close()
        _session = None


# ---------------------------------------------------------------------------
# 重试（仅 429 / 503 等可恢复错误）
# ---------------------------------------------------------------------------


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, DeepSeekRateLimitError):
        return True
    if isinstance(exc, DeepSeekServerError) and exc.status in {502, 503, 504}:
        return True
    # 网络抖动 / 超时：可重试
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    if isinstance(exc, aiohttp.ClientError):
        return True
    return False


async def _with_retry(coro_factory, *, max_retries: int, base_delay: float):
    """
    对可重试异常做指数退避（含少量抖动）。
    coro_factory: 无参异步可调用，每次重试重新发起请求。
    """
    last_exc: BaseException | None = None
    attempts = max(1, max_retries + 1)
    for attempt in range(attempts):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            if not _retryable(exc) or attempt >= attempts - 1:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, 0.25)
            logger.warning(
                "deepseek_retry",
                extra={
                    "event": "deepseek_retry",
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "delay_s": round(delay, 3),
                    "error": str(exc),
                    "status": getattr(exc, "status", None),
                },
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# 核心请求
# ---------------------------------------------------------------------------


def _raise_for_status(status: int, body: Any) -> None:
    msg = f"DeepSeek API 错误 {status}: {body}"
    if status in (401, 403):
        raise DeepSeekAuthError(msg, status=status, body=body)
    if status == 429:
        raise DeepSeekRateLimitError(msg, status=status, body=body)
    if status >= 500:
        raise DeepSeekServerError(msg, status=status, body=body)
    raise DeepSeekError(msg, status=status, body=body)


def _parse_result(
    body: dict[str, Any], fallback_model: str, mode: DeepSeekMode
) -> DeepSeekChatResult:
    try:
        choice0 = body["choices"][0]
        message = choice0["message"]
        content = (message.get("content") or "").strip()
        reasoning = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or choice0.get("reasoning_content")
            or ""
        )
        if isinstance(reasoning, str):
            reasoning = reasoning.strip()
        else:
            reasoning = str(reasoning or "")

        usage = body.get("usage") or {}
        finish_reason = str(choice0.get("finish_reason") or "")

        # reasoner 配额常被思考链占满 → content 为空；明确报错便于上层兜底/重试
        if not content:
            hint = f"finish_reason={finish_reason or '?'}"
            if reasoning:
                hint += f", reasoning_len={len(reasoning)}"
            raise DeepSeekResponseError(
                f"DeepSeek 返回空 content（{hint}）。"
                f"若使用 reasoner，请增大 max_tokens（建议>={settings.DEEPSEEK_REASONER_MIN_TOKENS}）",
                body=body,
            )

        return DeepSeekChatResult(
            content=content,
            reasoning_content=reasoning,
            model=str(body.get("model") or fallback_model),
            mode=mode,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            raw=body if isinstance(body, dict) else {},
        )
    except DeepSeekResponseError:
        raise
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
        raise DeepSeekResponseError(f"DeepSeek 返回格式异常: {body}", body=body) from exc


def _clamp_max_tokens(mode: DeepSeekMode, max_tokens: int) -> int:
    tokens = max(1, int(max_tokens))
    if mode == "reasoner":
        floor = int(getattr(settings, "DEEPSEEK_REASONER_MIN_TOKENS", 1024) or 1024)
        if tokens < floor:
            logger.info(
                "deepseek_raise_max_tokens",
                extra={
                    "event": "deepseek_raise_max_tokens",
                    "mode": mode,
                    "requested": tokens,
                    "applied": floor,
                },
            )
            return floor
    return tokens


def _build_payload(
    *,
    model: str,
    mode: DeepSeekMode,
    user_prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """
    reasoner 不支持 temperature / top_p 等采样参数（官方会忽略或报错），故按模式组装。
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "stream": False,
    }
    if mode == "chat":
        payload["temperature"] = temperature
    return payload


async def _deepseek_chat_once(
    *,
    user_prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    mode: DeepSeekMode,
) -> DeepSeekChatResult:
    if not settings.DEEPSEEK_API_KEY:
        raise DeepSeekAuthError("未配置 DEEPSEEK_API_KEY，请在项目根目录 .env 中设置")

    model = resolve_model(mode)
    max_tokens = _clamp_max_tokens(mode, max_tokens)
    url = f"{settings.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = _build_payload(
        model=model,
        mode=mode,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    started = time.perf_counter()
    session = await get_http_session()
    try:
        async with session.post(url, headers=headers, json=payload) as resp:
            try:
                body = await resp.json(content_type=None)
            except Exception as exc:
                text = await resp.text()
                raise DeepSeekResponseError(
                    f"DeepSeek 响应非 JSON（status={resp.status}）: {text[:300]}",
                    status=resp.status,
                    body=text[:500],
                ) from exc
            if resp.status >= 400:
                _raise_for_status(resp.status, body)
            result = _parse_result(body if isinstance(body, dict) else {}, model, mode)
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise DeepSeekServerError(f"DeepSeek 请求超时: {exc}", status=504) from exc
    except aiohttp.ClientError as exc:
        raise DeepSeekServerError(f"DeepSeek 网络错误: {exc}", status=503) from exc

    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "deepseek_chat_ok",
        extra={
            "event": "deepseek_chat_ok",
            "model": result.model,
            "mode": mode,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "latency_ms": round(latency_ms, 2),
            "has_reasoning": bool(result.reasoning_content),
        },
    )
    return result


async def deepseek_chat(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.2,
    max_tokens: int = 512,
    mode: DeepSeekMode | str | None = None,
) -> DeepSeekChatResult:
    """
    调用 DeepSeek Chat Completions。

    mode:
      - \"chat\"：快速对话（默认）
      - \"reasoner\"：深度推理（返回 reasoning_content）
      - None：使用 settings.DEEPSEEK_DEFAULT_MODE
    """
    resolved = resolve_mode(mode)
    max_retries = int(getattr(settings, "DEEPSEEK_MAX_RETRIES", 2))
    base_delay = float(getattr(settings, "DEEPSEEK_RETRY_BASE_DELAY", 0.8))

    return await _with_retry(
        lambda: _deepseek_chat_once(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            mode=resolved,
        ),
        max_retries=max_retries,
        base_delay=base_delay,
    )


async def deepseek_reasoner(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
) -> DeepSeekChatResult:
    """便捷封装：强制 reasoner 模式（忽略 temperature）。"""
    return await deepseek_chat(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        mode="reasoner",
    )
