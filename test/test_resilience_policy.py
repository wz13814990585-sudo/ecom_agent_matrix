from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ecom_agent_matrix.platform.resilience.retry import RetryPolicy, skill_retry_allowed
from ecom_agent_matrix.platform.resilience.timeout import DependencyTimeoutError, run_with_timeout


def test_read_transient_retry_is_bounded_and_backoff_capped():
    attempts = 0
    delays = []

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError()
        return "ok"

    policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.5, max_delay_seconds=0.75, jitter_seconds=0)

    async def scenario():
        with patch("ecom_agent_matrix.platform.resilience.retry.asyncio.sleep", new=AsyncMock()) as sleep:
            result = await policy.run(
                operation, retry_if=lambda exc: isinstance(exc, TimeoutError),
                on_retry=lambda _exc, _attempt, delay: delays.append(delay),
            )
        return result, sleep

    result, sleep = asyncio.run(scenario())
    assert result == "ok" and attempts == 3 and sleep.await_count == 2
    assert delays == [0.5, 0.75]


def test_write_and_approval_operations_are_not_retry_eligible():
    assert skill_retry_allowed(read_only=True, idempotent=True, side_effect=False)
    assert not skill_retry_allowed(read_only=False, idempotent=False, side_effect=True)
    assert not skill_retry_allowed(read_only=False, idempotent=True, side_effect=True)


def test_timeout_is_safe_and_cancellation_is_re_raised():
    async def timeout_case():
        with pytest.raises(DependencyTimeoutError, match="TIMEOUT"):
            await run_with_timeout(asyncio.sleep(1), 0.001)

    async def cancellation_case():
        task = asyncio.create_task(run_with_timeout(asyncio.sleep(10), 20))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(timeout_case())
    asyncio.run(cancellation_case())

