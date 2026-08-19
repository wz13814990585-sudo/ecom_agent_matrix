from __future__ import annotations

import asyncio
import inspect
from copy import deepcopy
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.modules.rag.lexical import lexical_search
from ecom_agent_matrix.modules.rag.retriever import (
    _cache_key,
    _hybrid_retrieve_detailed_uncached,
    _vector_channel,
    candidate_limit,
    rrf_fuse,
)


def _doc(source_id: str, text: str) -> dict:
    return {"source_id": source_id, "chunk_text": text}


def test_candidate_limit_is_bounded():
    assert candidate_limit(1) == 20
    assert candidate_limit(20) == 80
    assert candidate_limit(1000) == 80


def test_lexical_is_parameterized_bounded_and_embedding_independent():
    captured = {}

    async def execute(sql, params):
        captured.update(sql=sql, params=params)
        return [("SKU-1", "refund policy thirty days", {"doc_id": "D1"}, 2)]

    with patch(
        "ecom_agent_matrix.modules.rag.lexical.AsyncPGClient.execute_sql",
        new=execute,
    ):
        result = asyncio.run(lexical_search("refund policy", "en", None, 500))

    assert result[0]["bm25_score"] is not None
    assert "refund policy" not in captured["sql"]
    assert "ILIKE %s" in captured["sql"]
    assert "WHERE" in captured["sql"] and "LIMIT %s" in captured["sql"]
    assert captured["params"][-1] == 80
    source = inspect.getsource(lexical_search)
    assert "get_text_embedding" not in source and "vector_search" not in source


def test_vector_channel_does_not_depend_on_lexical():
    with patch(
        "ecom_agent_matrix.modules.rag.retriever.get_text_embedding",
        new=AsyncMock(return_value=[0.1]),
    ), patch(
        "ecom_agent_matrix.modules.rag.retriever.vector_search",
        new=AsyncMock(return_value=[{"chunk_text": "vector", "vector_score": 0.9}]),
    ):
        documents, _ = asyncio.run(_vector_channel("q", "en", None, 20))
    assert documents[0]["vector_score"] == 0.9


def test_channels_run_concurrently_and_can_return_disjoint_documents():
    entered: set[str] = set()
    both_entered = asyncio.Event()

    async def channel(name, document):
        entered.add(name)
        if len(entered) == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=0.2)
        return [document], 1.0

    async def rerank(_query, candidates, top_k):
        return candidates[:top_k], "keyword_fallback"

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.rag.retriever._vector_channel",
            new=lambda *_args: channel("vector", _doc("V", "vector only")),
        ), patch(
            "ecom_agent_matrix.modules.rag.retriever._lexical_channel",
            new=lambda *_args: channel("lexical", _doc("L", "lexical only")),
        ), patch(
            "ecom_agent_matrix.modules.rag.retriever.rerank_documents_detailed",
            new=rerank,
        ):
            return await _hybrid_retrieve_detailed_uncached("q", "en", top_k=8)

    result = asyncio.run(scenario())
    assert result.mode == "hybrid"
    assert {item["source_id"] for item in result.raw_documents} == {"V", "L"}


def test_vector_embedding_failure_degrades_to_lexical_only():
    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.rag.retriever.get_text_embedding",
            new=AsyncMock(side_effect=RuntimeError("model unavailable")),
        ), patch(
            "ecom_agent_matrix.modules.rag.retriever._lexical_channel",
            new=AsyncMock(return_value=([_doc("L", "refund")], 1.0)),
        ), patch(
            "ecom_agent_matrix.modules.rag.retriever.rerank_documents_detailed",
            new=AsyncMock(return_value=([_doc("L", "refund")], "keyword_fallback")),
        ):
            return await _hybrid_retrieve_detailed_uncached("refund", "en")

    result = asyncio.run(scenario())
    assert result.success and result.degraded and result.mode == "lexical_only"
    assert result.channel_errors == {"vector": "EMBEDDING_ERROR"}


def test_lexical_failure_degrades_to_vector_only():
    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.rag.retriever._vector_channel",
            new=AsyncMock(return_value=([_doc("V", "refund")], 1.0)),
        ), patch(
            "ecom_agent_matrix.modules.rag.retriever._lexical_channel",
            new=AsyncMock(side_effect=RuntimeError("db unavailable")),
        ), patch(
            "ecom_agent_matrix.modules.rag.retriever.rerank_documents_detailed",
            new=AsyncMock(return_value=([_doc("V", "refund")], "keyword_fallback")),
        ):
            return await _hybrid_retrieve_detailed_uncached("refund", "en")

    result = asyncio.run(scenario())
    assert result.success and result.degraded and result.mode == "vector_only"
    assert result.channel_errors == {"lexical": "RuntimeError"}


def test_both_channel_failures_return_failure_without_raw_errors():
    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.rag.retriever._vector_channel",
            new=AsyncMock(side_effect=RuntimeError("secret vector details")),
        ), patch(
            "ecom_agent_matrix.modules.rag.retriever._lexical_channel",
            new=AsyncMock(side_effect=ValueError("secret lexical details")),
        ):
            return await _hybrid_retrieve_detailed_uncached("q", "en")

    result = asyncio.run(scenario())
    assert not result.success and result.mode == "none"
    assert result.channel_errors == {"vector": "RuntimeError", "lexical": "ValueError"}
    assert "secret" not in str(result.channel_errors)


def test_rrf_uses_stable_identity_deduplicates_and_does_not_mutate_inputs():
    vector = [_doc("A", "same"), _doc("SHARED", "vector text")]
    lexical = [_doc("B", "same"), _doc("SHARED", "lexical text")]
    before_vector, before_lexical = deepcopy(vector), deepcopy(lexical)
    fused = rrf_fuse(vector, lexical)

    assert {item["source_id"] for item in fused} == {"A", "B", "SHARED"}
    assert len([item for item in fused if item["source_id"] == "SHARED"]) == 1
    assert vector == before_vector and lexical == before_lexical
    assert all("rrf_score" in item for item in fused)


def test_cache_and_logs_use_hybrid_v2_without_raw_query():
    assert settings.RAG_RETRIEVAL_VERSION == "hybrid-v2"
    new_key = _cache_key("private customer query", "en", None, 8)
    with patch.object(settings, "RAG_RETRIEVAL_VERSION", "hybrid-v1"):
        old_key = _cache_key("private customer query", "en", None, 8)
    assert new_key != old_key
    import ecom_agent_matrix.modules.rag.retriever as retriever

    source = inspect.getsource(retriever)
    assert '"query": query' not in source

