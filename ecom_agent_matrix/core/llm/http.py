"""LLM HTTP 基础设施：进程内复用 session + 可恢复错误重试。"""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

from ecom_agent_matrix.core.llm.types import LLMRateLimitError, LLMServerError
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.platform.observability.metrics import metrics
from ecom_agent_matrix.platform.resilience.retry import RetryPolicy

logger = setup_logger("llm.http")

_session: aiohttp.ClientSession | None = None
_session_lock = asyncio.Lock()


async def get_http_session() -> aiohttp.ClientSession:
    """获取进程内单例 aiohttp session（懒创建）。超时在每次请求上设置。"""
    global _session
    if _session is not None and not _session.closed:
        return _session
    async with _session_lock:
        if _session is None or _session.closed:
            _session = aiohttp.ClientSession()
    return _session


async def close_http_session() -> None:
    """进程退出时关闭 session。"""
    global _session
    async with _session_lock:
        if _session is not None and not _session.closed:
            await _session.close()
        _session = None


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, LLMRateLimitError):
        return True
    if isinstance(exc, LLMServerError) and exc.status in {502, 503, 504}:
        return True
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    if isinstance(exc, aiohttp.ClientError):
        return True
    return False


async def with_retry(
    coro_factory,
    *,
    max_retries: int,
    base_delay: float,
    extra: dict[str, Any] | None = None,
):
    """
    对可重试异常做指数退避（含少量抖动）。
    coro_factory: 无参异步可调用，每次重试重新发起请求。
    """
    component = str((extra or {}).get("provider") or "llm")

    def on_retry(exc: BaseException, attempt: int, delay: float) -> None:
        reason = (
            "429" if isinstance(exc, LLMRateLimitError)
            else "timeout" if isinstance(exc, (asyncio.TimeoutError, TimeoutError))
            else "connection" if isinstance(exc, aiohttp.ClientError)
            else "5xx"
        )
        metrics.external_retries.labels(f"llm:{component}", reason).inc()
        payload = {
            "event": "llm_retry",
            "attempt": attempt,
            "max_retries": max_retries,
            "delay_s": round(delay, 3),
            "error_type": type(exc).__name__,
            "component": f"llm:{component}",
            "status": getattr(exc, "status", None),
        }
        if extra:
            payload.update(extra)
        logger.warning("llm_retry", extra=payload)

    return await RetryPolicy(
        max_attempts=max(1, int(max_retries) + 1),
        base_delay_seconds=float(base_delay),
        max_delay_seconds=max(float(base_delay), float(base_delay) * 4),
        jitter_seconds=0.25,
    ).run(coro_factory, retry_if=is_retryable, on_retry=on_retry)
