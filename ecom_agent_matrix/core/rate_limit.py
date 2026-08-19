"""进程内 / Redis 分布式限流（Semaphore 语义）。"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.db.redis_client import AsyncRedisClient

logger = setup_logger("core.rate_limit")

_process_semaphores: dict[str, asyncio.Semaphore] = {}


def _process_semaphore(name: str, limit: int) -> asyncio.Semaphore:
    if name not in _process_semaphores:
        _process_semaphores[name] = asyncio.Semaphore(max(1, int(limit)))
    return _process_semaphores[name]


@asynccontextmanager
async def acquire_slot(
    name: str,
    *,
    limit: int,
    mode: str | None = None,
    ttl_sec: float = 60.0,
) -> AsyncIterator[str]:
    """
    获取一个并发槽位。
    - process: 进程内 Semaphore
    - redis: Redis 计数器（多实例共享）；Redis 不可用时回退 process
    yield 值为后端标识（process|redis|process_fallback）
    """
    resolved = (mode or getattr(settings, "SOCIAL_RATE_LIMIT_MODE", "process") or "process").lower()
    if resolved != "redis":
        async with _process_semaphore(name, limit):
            yield "process"
        return

    token = f"{uuid.uuid4().hex}"
    key = f"rate:{name}:slots"
    acquired = False
    try:
        redis = await AsyncRedisClient.get_client()
        # 简单令牌集合：成员数 < limit 则可入
        deadline = time.monotonic() + max(0.5, ttl_sec)
        while time.monotonic() < deadline:
            count = await redis.scard(key)
            if count < limit:
                added = await redis.sadd(key, token)
                if added:
                    await redis.expire(key, int(max(ttl_sec, 5)))
                    acquired = True
                    break
            await asyncio.sleep(0.05)
        if not acquired:
            # 排队超时：仍用进程内槽位兜底，避免任务永久卡住
            logger.warning(
                "redis_rate_limit_timeout",
                extra={"event": "redis_rate_limit_timeout", "agent": name},
            )
            async with _process_semaphore(name, limit):
                yield "process_fallback"
            return
        yield "redis"
    except Exception as exc:
        logger.warning(
            "redis_rate_limit_fallback",
            extra={"event": "redis_rate_limit_fallback", "agent": name, "error_type": type(exc).__name__},
        )
        async with _process_semaphore(name, limit):
            yield "process_fallback"
    finally:
        if acquired:
            try:
                redis = await AsyncRedisClient.get_client()
                await redis.srem(key, token)
            except Exception:
                pass
