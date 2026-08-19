from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.core.memory.long_vector_memory import AgentLongVectorMemory
from ecom_agent_matrix.core.memory.short_memory import AgentShortMemory
from ecom_agent_matrix.core.security import SecurityContext


def _security(tenant="tenant-a", user="user-a", store="store-a"):
    return SecurityContext(
        subject=user, user_id=user, tenant_id=tenant, store_id=store,
        roles=frozenset({"viewer"}), scopes=frozenset(), auth_type="jwt", authenticated=True,
    )


def test_short_memory_keys_isolate_tenant_and_user_and_keep_legacy_namespace():
    first = AgentShortMemory("session-x", tenant_id="tenant-a", user_id="user-a")
    other_tenant = AgentShortMemory("session-x", tenant_id="tenant-b", user_id="user-a")
    other_user = AgentShortMemory("session-x", tenant_id="tenant-a", user_id="user-b")
    assert len({first.key, other_tenant.key, other_user.key}) == 3
    assert "tenant-a" not in first.key and "user-a" not in first.key
    assert AgentShortMemory("session-x").key == "agent:short_mem:session-x"


def test_long_memory_save_forces_trusted_tenant_and_store_metadata():
    captured = {}

    async def execute(sql, params, **kwargs):
        captured.update(sql=sql, params=params)
        return [[7]]

    async def scenario():
        with patch(
            "ecom_agent_matrix.core.memory.long_vector_memory.get_text_embedding",
            new=AsyncMock(return_value=[0.1]),
        ), patch(
            "ecom_agent_matrix.core.memory.long_vector_memory.AsyncPGClient.execute_write",
            new=execute,
        ):
            return await AgentLongVectorMemory().save_memory(
                "agent", "content",
                {"tenant_id": "fake", "store_id": "fake", "success": True},
                context=_security(),
            )

    assert asyncio.run(scenario()) == 7
    assert captured["params"][:2] == ["tenant-a", "store-a"]
    metadata = json.loads(captured["params"][5])
    assert metadata["tenant_id"] == "tenant-a"
    assert metadata["store_id"] == "store-a"


def test_long_memory_recall_caller_cannot_override_trusted_scope():
    captured = {}

    async def execute(sql, params, **kwargs):
        captured.update(sql=sql, params=params)
        return []

    async def scenario():
        with patch(
            "ecom_agent_matrix.core.memory.long_vector_memory.get_text_embedding",
            new=AsyncMock(return_value=[0.1]),
        ), patch(
            "ecom_agent_matrix.core.memory.long_vector_memory.AsyncPGClient.execute_read",
            new=execute,
        ):
            return await AgentLongVectorMemory().recall(
                "query", "agent", meta_filter={"tenant_id": "fake", "store_id": "fake"},
                context=_security(),
            )

    asyncio.run(scenario())
    assert "tenant-a" in captured["params"] and "store-a" in captured["params"]
    assert "fake" not in captured["params"]


def test_long_memory_deprecate_is_physically_tenant_scoped():
    captured = {}

    async def execute(sql, params, **kwargs):
        captured.update(sql=sql, params=params, scope=kwargs["scope"])
        return [[11]]

    async def scenario():
        with patch(
            "ecom_agent_matrix.core.memory.long_vector_memory.AsyncPGClient.execute_write",
            new=execute,
        ):
            return await AgentLongVectorMemory().deprecate_memory(11, context=_security())

    assert asyncio.run(scenario()) is True
    assert "tenant_id = %s AND store_id = %s" in captured["sql"]
    assert captured["params"] == [11, "tenant-a", "store-a"]
    assert captured["scope"].usable
