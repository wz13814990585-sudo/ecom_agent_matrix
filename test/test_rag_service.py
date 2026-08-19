from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from ecom_agent_matrix.modules.rag.schemas import HybridRetrievalResult, RAGDocument, RAGRequest
from ecom_agent_matrix.modules.rag.service import RAGService


def test_rag_request_validation_boundaries():
    assert RAGRequest(query="  refund policy  ").query == "refund policy"
    with pytest.raises(ValidationError):
        RAGRequest(query="   ")
    with pytest.raises(ValidationError):
        RAGRequest(query="x", top_k=21)
    with pytest.raises(ValidationError):
        RAGRequest(query="x", price_max=-1)


def test_retrieve_returns_typed_documents_and_citations():
    async def scenario():
        raw = [
            {
                "sku": "SKU-1",
                "chunk_text": "Refunds are available within 30 days.",
                "meta": {"document_id": "refund-policy"},
                "score": 0.9,
            }
        ]
        with patch(
            "ecom_agent_matrix.modules.rag.service.hybrid_retrieve_detailed",
            new=AsyncMock(return_value=HybridRetrievalResult(
                success=True, raw_documents=raw, mode="hybrid", cached=True,
                latency_ms=12.5,
            )),
        ):
            return await RAGService().retrieve(RAGRequest(query="refund", task_id="T1"))

    result = asyncio.run(scenario())
    assert result.success and result.cached
    assert result.recall_count == 1
    assert isinstance(result.documents[0], RAGDocument)
    assert result.documents[0].source_id == "refund-policy"
    assert result.citations[0].citation_id == "S1"


def test_retrieval_error_is_safe_and_structured():
    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.rag.service.hybrid_retrieve_detailed",
            new=AsyncMock(
                side_effect=RuntimeError("postgres://user:password@secret-host/private.sql")
            ),
        ):
            return await RAGService().retrieve(RAGRequest(query="refund", task_id="T1"))

    result = asyncio.run(scenario())
    assert result.success is False
    assert result.error_code == "RETRIEVAL_ERROR"
    assert result.error_msg == "RAG retrieval failed"
    assert "password" not in result.error_msg
    assert "secret-host" not in result.error_msg


def test_service_module_has_no_mcp_dependency():
    import ecom_agent_matrix.modules.rag.service as service_module

    source = inspect.getsource(service_module)
    assert "MCPMessage" not in source
    assert "mcp_bus" not in source
    assert "register_agent" not in source
    assert "build_reply" not in source
