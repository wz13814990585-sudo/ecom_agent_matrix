from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.modules.rag.formatter import normalize_documents, stable_source_id
from ecom_agent_matrix.modules.rag.schemas import RAGRequest
from ecom_agent_matrix.modules.rag.service import RAGService


def test_fallback_source_identity_is_deterministic_and_ranked():
    raw = [
        {"sku": "A", "chunk_text": "alpha"},
        {"sku": "B", "chunk_text": "beta"},
    ]
    first = normalize_documents(raw)
    second = normalize_documents(raw)
    assert [document.citation_id for document in first] == ["S1", "S2"]
    assert first[0].source_id == second[0].source_id == stable_source_id(raw[0])
    assert first[0].source_id.startswith("sha256:")


def test_valid_citation_is_grounded():
    async def scenario():
        docs = [{"chunk_text": "Returns are accepted.", "meta": {"doc_id": "D1"}}]
        with patch(
            "ecom_agent_matrix.modules.rag.service.hybrid_retrieve",
            new=AsyncMock(return_value=(docs, False, 1)),
        ), patch(
            "ecom_agent_matrix.modules.rag.service.llm_explain",
            new=AsyncMock(return_value=("Returns are accepted [S1].", "test", "")),
        ):
            return await RAGService().answer(RAGRequest(query="returns"))

    result = asyncio.run(scenario())
    assert result.grounded is True
    assert [citation.citation_id for citation in result.citations] == ["S1"]


def test_nonexistent_citation_is_not_legal_and_not_grounded():
    async def scenario():
        docs = [{"chunk_text": "Returns are accepted.", "meta": {"doc_id": "D1"}}]
        with patch(
            "ecom_agent_matrix.modules.rag.service.hybrid_retrieve",
            new=AsyncMock(return_value=(docs, False, 1)),
        ), patch(
            "ecom_agent_matrix.modules.rag.service.llm_explain",
            new=AsyncMock(return_value=("Invented claim [S99].", "test", "")),
        ):
            return await RAGService().answer(RAGRequest(query="returns"))

    result = asyncio.run(scenario())
    assert result.grounded is False
    assert "S99" not in {citation.citation_id for citation in result.citations}


def test_no_documents_is_not_grounded_and_skips_answer_llm():
    async def scenario():
        llm = AsyncMock()
        with patch(
            "ecom_agent_matrix.modules.rag.service.hybrid_retrieve",
            new=AsyncMock(return_value=([], False, 1)),
        ), patch(
            "ecom_agent_matrix.modules.rag.service.llm_explain", new=llm
        ):
            result = await RAGService().answer(RAGRequest(query="unknown"))
        return result, llm

    result, llm = asyncio.run(scenario())
    assert result.success is True
    assert result.grounded is False
    assert result.answer_source == "no_knowledge"
    llm.assert_not_awaited()


def test_generation_exception_uses_safe_grounded_fallback():
    async def scenario():
        docs = [{"chunk_text": "Returns are accepted.", "meta": {"doc_id": "D1"}}]
        with patch(
            "ecom_agent_matrix.modules.rag.service.hybrid_retrieve",
            new=AsyncMock(return_value=(docs, False, 1)),
        ), patch(
            "ecom_agent_matrix.modules.rag.service.llm_explain",
            new=AsyncMock(side_effect=RuntimeError("api_key=TOP_SECRET")),
        ):
            return await RAGService().answer(RAGRequest(query="returns"))

    result = asyncio.run(scenario())
    assert result.success is True
    assert result.grounded is True
    assert result.error_code == "GENERATION_ERROR"
    assert "TOP_SECRET" not in result.error_msg
