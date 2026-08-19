from __future__ import annotations

import asyncio

import pytest

from ecom_agent_matrix.platform.resilience.circuit_breaker import (
    CircuitBreaker, CircuitOpenError, CircuitState,
)


def test_circuit_closed_failures_open_fast_fail_and_successful_probe_closes():
    async def scenario():
        breaker = CircuitBreaker("test", failure_threshold=2, reset_seconds=0.01)
        assert breaker.state == CircuitState.CLOSED

        async def transient():
            raise TimeoutError()

        for _ in range(2):
            with pytest.raises(TimeoutError):
                await breaker.call(transient, is_transient=lambda exc: isinstance(exc, TimeoutError))
        assert breaker.state == CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            await breaker.call(lambda: asyncio.sleep(0), is_transient=lambda _exc: True)
        await asyncio.sleep(0.02)
        result = await breaker.call(lambda: asyncio.sleep(0, result="ok"), is_transient=lambda _exc: True)
        return breaker, result

    breaker, result = asyncio.run(scenario())
    assert result == "ok" and breaker.state == CircuitState.CLOSED


def test_non_transient_validation_failure_does_not_open_circuit():
    async def scenario():
        breaker = CircuitBreaker("validation", failure_threshold=1)
        async def invalid():
            raise ValueError("invalid")
        with pytest.raises(ValueError):
            await breaker.call(invalid, is_transient=lambda exc: isinstance(exc, TimeoutError))
        return breaker

    breaker = asyncio.run(scenario())
    assert breaker.state == CircuitState.CLOSED and breaker.failure_count == 0


def test_half_open_allows_only_one_probe_then_closes_on_success():
    async def scenario():
        breaker = CircuitBreaker("half-open", failure_threshold=1, reset_seconds=0.01)

        async def transient():
            raise TimeoutError()

        with pytest.raises(TimeoutError):
            await breaker.call(transient, is_transient=lambda exc: isinstance(exc, TimeoutError))
        await asyncio.sleep(0.02)

        entered = asyncio.Event()
        release = asyncio.Event()

        async def probe():
            entered.set()
            await release.wait()
            return "recovered"

        task = asyncio.create_task(breaker.call(probe, is_transient=lambda _exc: True))
        await entered.wait()
        assert breaker.state == CircuitState.HALF_OPEN
        with pytest.raises(CircuitOpenError):
            await breaker.call(lambda: asyncio.sleep(0), is_transient=lambda _exc: True)
        release.set()
        assert await task == "recovered"
        return breaker

    breaker = asyncio.run(scenario())
    assert breaker.state == CircuitState.CLOSED
