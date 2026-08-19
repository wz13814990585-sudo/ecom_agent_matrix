from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.core.security import SecurityContext, TenantScope
from ecom_agent_matrix.modules.rag.lexical import lexical_search
from ecom_agent_matrix.modules.rag.retriever import _cache_key, vector_search
from ecom_agent_matrix.modules.rag.schemas import RAGRequest
from ecom_agent_matrix.modules.rag.schemas import RAGRetrievalResult
from ecom_agent_matrix.modules.rag.service import RAGService
from ecom_agent_matrix.core.skill.base_skill import SkillResult
from ecom_agent_matrix.core.tasking import normalize_task_context
from ecom_agent_matrix.modules.agent_cluster.handlers.crm import run_crm_workflow


def _scope(tenant, store):
    return TenantScope(tenant_id=tenant, store_id=store, identity_trusted=True)


def _security(tenant="tenant-a", store="store-a"):
    return SecurityContext(
        subject="u", user_id="u", tenant_id=tenant, store_id=store,
        roles=frozenset({"viewer"}), scopes=frozenset(), auth_type="jwt", authenticated=True,
    )


def test_same_query_has_different_non_plaintext_cache_key_across_tenants():
    a = _cache_key("refund policy", "en", None, 5, _scope("tenant-a", "store-a"))
    b = _cache_key("refund policy", "en", None, 5, _scope("tenant-b", "store-b"))
    assert a != b
    assert "tenant-a" not in a and "store-a" not in a
    assert ":v2:" in a


def test_vector_and_lexical_sql_use_identical_trusted_scope():
    captured = []

    async def execute(sql, params, **kwargs):
        captured.append((sql, params, kwargs["scope"]))
        return []

    async def scenario():
        scope = _scope("tenant-a", "store-a")
        with patch(
            "ecom_agent_matrix.modules.rag.retriever.AsyncPGClient.execute_read", new=execute
        ), patch(
            "ecom_agent_matrix.modules.rag.lexical.AsyncPGClient.execute_read", new=execute
        ):
            await vector_search([0.1], "en", top_k=5, scope=scope)
            await lexical_search("refund policy", "en", None, 5, scope=scope)

    asyncio.run(scenario())
    assert len(captured) == 2
    for sql, params, scope in captured:
        assert "tenant_id = %s AND store_id = %s" in sql
        assert "tenant-a" in params and "store-a" in params
        assert scope == _scope("tenant-a", "store-a")


def test_rag_service_production_without_trusted_scope_fails_closed():
    async def scenario():
        with patch("ecom_agent_matrix.modules.rag.service.settings.APP_ENV", "production"), patch(
            "ecom_agent_matrix.modules.rag.service.hybrid_retrieve_detailed", new=AsyncMock()
        ) as retrieve:
            result = await RAGService().retrieve(RAGRequest(query="refund policy"))
        return result, retrieve

    result, retrieve = asyncio.run(scenario())
    assert not result.success and result.error_code == "RETRIEVAL_ERROR"
    retrieve.assert_not_awaited()


def test_rag_agent_scope_helper_ignores_payload_tenant_values():
    from ecom_agent_matrix.core.security import tenant_scope_from_security

    payload = {"tenant_id": "fake", "store_id": "fake"}
    scope = tenant_scope_from_security(_security())
    assert payload["tenant_id"] != scope.tenant_id
    assert (scope.tenant_id, scope.store_id, scope.identity_trusted) == (
        "tenant-a", "store-a", True
    )


def test_index_version_is_v2():
    from ecom_agent_matrix.config.settings import settings

    assert settings.RAG_INDEX_VERSION == "v2"


def test_crm_rag_uses_trusted_task_scope():
    retrieve = AsyncMock(return_value=RAGRetrievalResult(
        success=False, retrieval_version="hybrid-v2", error_code="RETRIEVAL_ERROR"
    ))
    reply = SkillResult(success=True, data={"answer": "safe", "llm_ok": True})
    ctx = normalize_task_context(
        {"query": "help me reply about refund policy", "use_rag": True},
        task_id="task-1", security=_security(),
    )

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.rag_service.retrieve", new=retrieve
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.exec_skill",
            new=AsyncMock(return_value=reply),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.AgentShortMemory.append",
            new=AsyncMock(),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.AgentShortMemory.get_all",
            new=AsyncMock(return_value=[]),
        ):
            return await run_crm_workflow(ctx)

    result = asyncio.run(scenario())
    assert result.success
    assert retrieve.await_args.kwargs["scope"] == _scope("tenant-a", "store-a")
