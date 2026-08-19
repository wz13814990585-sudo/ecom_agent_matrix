from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException, Request

from ecom_agent_matrix.api.main import health
from ecom_agent_matrix.core.security import SecurityContext
from ecom_agent_matrix.platform.resilience.rate_limit import (
    InProcessRateLimiter, enforce_business_rate_limit, rate_limiter,
)


def _security(tenant):
    return SecurityContext(
        subject="u", user_id="u", tenant_id=tenant, store_id="s",
        roles=frozenset({"viewer"}), scopes=frozenset(), auth_type="jwt", authenticated=True,
    )


def _request(path="/api/v1/tasks"):
    return Request({
        "type": "http", "method": "POST", "path": path, "query_string": b"",
        "headers": [], "server": ("test", 80), "scheme": "http",
        "route": SimpleNamespace(path=path),
    })


def test_in_process_limiter_allows_then_rejects_and_tenants_are_isolated():
    limiter = InProcessRateLimiter()

    async def scenario():
        first = await limiter.check("tenant-a:user", limit=2, window_seconds=60, now=1)
        second = await limiter.check("tenant-a:user", limit=2, window_seconds=60, now=2)
        rejected = await limiter.check("tenant-a:user", limit=2, window_seconds=60, now=3)
        other = await limiter.check("tenant-b:user", limit=2, window_seconds=60, now=3)
        return first, second, rejected, other

    first, second, rejected, other = asyncio.run(scenario())
    assert first[0] and second[0] and not rejected[0] and rejected[1] > 0 and other[0]


def test_dependency_returns_429_rate_limited_with_retry_after():
    async def scenario():
        await rate_limiter.clear()
        with patch("ecom_agent_matrix.platform.resilience.rate_limit.settings.RATE_LIMIT_ENABLED", True), patch(
            "ecom_agent_matrix.platform.resilience.rate_limit.settings.RATE_LIMIT_REQUESTS", 1
        ):
            await enforce_business_rate_limit(_request(), _security("tenant-a"))
            with pytest.raises(HTTPException) as raised:
                await enforce_business_rate_limit(_request(), _security("tenant-a"))
        return raised.value

    error = asyncio.run(scenario())
    assert error.status_code == 429 and error.detail == "RATE_LIMITED"
    assert int(error.headers["Retry-After"]) >= 1


def test_health_is_not_business_rate_limited_or_dependency_probed():
    with patch("ecom_agent_matrix.api.health.check_postgres", side_effect=AssertionError), patch(
        "ecom_agent_matrix.api.health.check_redis", side_effect=AssertionError
    ):
        assert "status" in asyncio.run(health())
