from __future__ import annotations

import pytest
from pydantic import ValidationError

from ecom_agent_matrix.core.security import SecurityContext
from ecom_agent_matrix.core.security import require_trusted_ingress
from ecom_agent_matrix.core.security.errors import AuthenticationError
from ecom_agent_matrix.core.tasking import normalize_task_context
from ecom_agent_matrix.config.constants import AGENT_QUERY
from ecom_agent_matrix.core.skill.skill_registry import (
    current_skill_execution_context,
    exec_skill,
    skill_execution_context,
)
import asyncio
import ecom_agent_matrix.modules.skills  # noqa: F401


def principal(**updates) -> SecurityContext:
    values = {
        "subject": "subject-1",
        "user_id": "user-1",
        "tenant_id": "tenant-1",
        "store_id": "store-1",
        "roles": frozenset({"viewer"}),
        "scopes": frozenset(),
        "auth_type": "jwt",
        "authenticated": True,
    }
    values.update(updates)
    return SecurityContext(**values)


def test_security_context_forbids_credentials_and_mutation():
    security = principal()
    assert security.roles == frozenset({"viewer"})
    assert "token" not in security.model_dump()
    with pytest.raises(ValidationError):
        SecurityContext(**security.model_dump(), raw_jwt="secret")
    with pytest.raises(ValidationError):
        security.user_id = "changed"


def test_trusted_security_identity_wins_and_reserved_params_are_removed():
    security = principal()
    ctx = normalize_task_context(
        {
            "query": "q",
            "tenant_id": "fake-tenant",
            "user_id": "fake-user",
            "store_id": "fake-store",
            "roles": ["admin"],
            "scope": "risk:write",
            "sku": "SKU-1",
        },
        task_id="root",
        security=security,
    )
    assert (ctx.tenant_id, ctx.user_id, ctx.store_id) == (
        "tenant-1", "user-1", "store-1"
    )
    assert ctx.identity_trusted is True
    assert ctx.params == {"query": "q", "sku": "SKU-1"}


def test_legacy_task_context_identity_remains_compatible_but_untrusted():
    ctx = normalize_task_context(
        {"query": "q", "tenant_id": "legacy-t", "user_id": "legacy-u"}
    )
    assert ctx.tenant_id == "legacy-t" and ctx.user_id == "legacy-u"
    assert ctx.identity_trusted is False


def test_skill_context_receives_identity_and_fake_params_cannot_elevate():
    security = principal(roles=frozenset({"admin"}), scopes=frozenset({"risk:write"}))
    ctx = normalize_task_context({"query": "q"}, task_id="root", security=security)
    with skill_execution_context(AGENT_QUERY, task_context=ctx, security=security):
        current = current_skill_execution_context()
        assert current is not None
        assert (current.task_id, current.tenant_id, current.user_id, current.store_id) == (
            "root", "tenant-1", "user-1", "store-1"
        )
        assert current.roles == frozenset({"admin"}) and current.identity_trusted
        result = asyncio.run(exec_skill(
            "record_competitor_price",
            {
                "target_sku": "SKU-1", "competitor": "Temu", "compete_price": 10,
                "roles": ["admin"],
            },
        ))
    assert result.error_code == "PERMISSION_DENIED"


def test_production_mcp_ingress_requires_trusted_security():
    with pytest.raises(AuthenticationError):
        require_trusted_ingress(None, app_env="production")
    require_trusted_ingress(principal(), app_env="production")
    require_trusted_ingress(None, app_env="development")
