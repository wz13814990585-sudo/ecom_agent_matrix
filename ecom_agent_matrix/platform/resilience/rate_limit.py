"""Single-process demo API limiter keyed by hashed authenticated principal."""
from __future__ import annotations

import asyncio
from collections import deque
import math
import time

from fastapi import Depends, HTTPException, Request, status

from ecom_agent_matrix.api.auth import get_current_security_context
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.security import SecurityContext
from ecom_agent_matrix.platform.observability.context import identity_hash
from ecom_agent_matrix.platform.observability.metrics import metrics


class InProcessRateLimiter:
    def __init__(self):
        self._buckets: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str, *, limit: int, window_seconds: float, now: float | None = None) -> tuple[bool, int]:
        current = time.monotonic() if now is None else float(now)
        window = max(0.01, float(window_seconds))
        async with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            while bucket and current - bucket[0] >= window:
                bucket.popleft()
            if len(bucket) >= max(1, int(limit)):
                retry_after = max(1, math.ceil(window - (current - bucket[0])))
                return False, retry_after
            bucket.append(current)
            return True, 0

    async def clear(self) -> None:
        async with self._lock:
            self._buckets.clear()


rate_limiter = InProcessRateLimiter()


async def enforce_business_rate_limit(
    request: Request,
    security: SecurityContext = Depends(get_current_security_context),
) -> None:
    if not bool(settings.RATE_LIMIT_ENABLED):
        return
    route = getattr(request.scope.get("route"), "path", request.url.path)
    key = f"{identity_hash(security.tenant_id)}:{identity_hash(security.user_id)}"
    allowed, retry_after = await rate_limiter.check(
        key,
        limit=int(settings.RATE_LIMIT_REQUESTS),
        window_seconds=float(settings.RATE_LIMIT_WINDOW_SECONDS),
    )
    if not allowed:
        metrics.rate_limit_rejections.labels(route).inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="RATE_LIMITED",
            headers={"Retry-After": str(retry_after)},
        )


__all__ = ["InProcessRateLimiter", "enforce_business_rate_limit", "rate_limiter"]
