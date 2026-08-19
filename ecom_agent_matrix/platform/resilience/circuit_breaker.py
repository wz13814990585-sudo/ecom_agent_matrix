"""Small process-local circuit breaker for bounded external components."""
from __future__ import annotations

import asyncio
from enum import Enum
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitOpenError(RuntimeError):
    error_code = "CIRCUIT_OPEN"


class CircuitBreaker:
    def __init__(self, name: str, *, failure_threshold: int = 5, reset_seconds: float = 30.0):
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.reset_seconds = max(0.01, float(reset_seconds))
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.opened_at = 0.0
        self._probe_in_flight = False
        self._lock = asyncio.Lock()

    async def _allow(self) -> None:
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if time.monotonic() - self.opened_at < self.reset_seconds:
                    raise CircuitOpenError("CIRCUIT_OPEN")
                self.state = CircuitState.HALF_OPEN
            if self.state == CircuitState.HALF_OPEN:
                if self._probe_in_flight:
                    raise CircuitOpenError("CIRCUIT_OPEN")
                self._probe_in_flight = True

    async def call(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        is_transient: Callable[[BaseException], bool],
    ) -> T:
        await self._allow()
        try:
            result = await operation()
        except asyncio.CancelledError:
            await self._finish_probe()
            raise
        except Exception as exc:
            if is_transient(exc):
                await self._record_failure()
            else:
                await self._finish_probe()
            raise
        await self._record_success()
        return result

    async def _record_failure(self) -> None:
        async with self._lock:
            self._probe_in_flight = False
            self.failure_count += 1
            if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()

    async def _record_success(self) -> None:
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.opened_at = 0.0
            self._probe_in_flight = False

    async def _finish_probe(self) -> None:
        async with self._lock:
            self._probe_in_flight = False


_BREAKERS: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, *, failure_threshold: int = 5, reset_seconds: float = 30.0) -> CircuitBreaker:
    if name not in _BREAKERS:
        _BREAKERS[name] = CircuitBreaker(
            name, failure_threshold=failure_threshold, reset_seconds=reset_seconds
        )
    return _BREAKERS[name]


__all__ = ["CircuitBreaker", "CircuitOpenError", "CircuitState", "get_circuit_breaker"]

