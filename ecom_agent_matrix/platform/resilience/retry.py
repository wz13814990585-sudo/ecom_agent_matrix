"""Bounded retry policy for explicitly safe transient operations."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 2.0
    jitter_seconds: float = 0.25

    def delay_for(self, retry_index: int) -> float:
        bounded = min(
            max(0.0, self.base_delay_seconds) * (2 ** max(0, retry_index)),
            max(0.0, self.max_delay_seconds),
        )
        return bounded + random.uniform(0, max(0.0, self.jitter_seconds))

    async def run(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        retry_if: Callable[[BaseException], bool],
        on_retry: Callable[[BaseException, int, float], None] | None = None,
    ) -> T:
        attempts = max(1, int(self.max_attempts))
        for attempt in range(attempts):
            try:
                return await operation()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempt >= attempts - 1 or not retry_if(exc):
                    raise
                delay = self.delay_for(attempt)
                if on_retry:
                    on_retry(exc, attempt + 1, delay)
                await asyncio.sleep(delay)
        raise RuntimeError("retry exhausted")


def skill_retry_allowed(*, read_only: bool, idempotent: bool, side_effect: bool) -> bool:
    return bool(read_only and idempotent and not side_effect)


__all__ = ["RetryPolicy", "skill_retry_allowed"]

