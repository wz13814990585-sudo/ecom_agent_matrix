from __future__ import annotations

import asyncio
from typing import Awaitable, TypeVar

T = TypeVar("T")


class DependencyTimeoutError(RuntimeError):
    error_code = "TIMEOUT"


async def run_with_timeout(awaitable: Awaitable[T], seconds: float) -> T:
    try:
        return await asyncio.wait_for(awaitable, timeout=max(0.001, float(seconds)))
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError as exc:
        raise DependencyTimeoutError("TIMEOUT") from exc


__all__ = ["DependencyTimeoutError", "run_with_timeout"]

