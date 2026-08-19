from __future__ import annotations

import asyncio
import contextlib
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ecom_agent_matrix.config.constants import AGENT_MASTER, AGENT_RAG
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.skill.base_skill import SkillResult
from ecom_agent_matrix.modules.agent_cluster.handlers.crm import run_crm_workflow
from ecom_agent_matrix.modules.rag.rag_agent import rag_agent
from ecom_agent_matrix.modules.rag.schemas import (
    RAGAnswerResult,
    RAGCitation,
    RAGDocument,
    RAGRetrievalResult,
)


def _skill_reply() -> SkillResult:
    return SkillResult(
        success=True,
        data={
            "answer": "answer",
            "llm_ok": True,
            "rag_used": False,
            "rag_doc_count": 0,
            "rag_error": "",
            "citations": [],
        },
    )


def _retrieval(*, success=True) -> RAGRetrievalResult:
    document = RAGDocument(
        citation_id="S1",
        source_id="policy-1",
        title="Refund policy",
        chunk_text="Refunds are accepted within 30 days.",
    )
    citation = RAGCitation(
        citation_id="S1",
        source_id="policy-1",
        title="Refund policy",
    )
    return RAGRetrievalResult(
        success=success,
        documents=[document] if success else [],
        citations=[citation] if success else [],
        recall_count=1 if success else 0,
        retrieval_version="hybrid-v1",
        error_code="" if success else "RETRIEVAL_ERROR",
        error_msg="" if success else "RAG retrieval failed",
    )


def test_static_dependency_direction_is_enforced():
    rag_dir = Path("ecom_agent_matrix/modules/rag")
    combined = "\n".join(path.read_text() for path in rag_dir.glob("*.py"))
    assert "modules.skills.crm_reply" not in combined

    import ecom_agent_matrix.modules.rag.rag_agent as agent_module
    import ecom_agent_matrix.modules.skills.crm_reply as crm_reply_module

    agent_source = inspect.getsource(agent_module)
    skill_source = inspect.getsource(crm_reply_module)
    assert "hybrid_retrieve" not in agent_source
    assert "modules.rag.retriever" not in skill_source
    assert "hybrid_retrieve" not in skill_source


@pytest.mark.parametrize(
    ("query", "explicit", "expected_calls"),
    [
        ("hello", True, 1),
        ("这个商品的材质是什么", None, 1),
        ("这个商品的材质是什么", False, 0),
    ],
)
def test_crm_workflow_owns_retrieval_policy(query, explicit, expected_calls):
    async def scenario():
        retrieve = AsyncMock(return_value=_retrieval())
        payload = {"query": query, "use_rag": explicit}
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.rag_service.retrieve",
            new=retrieve,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.AgentShortMemory",
            side_effect=RuntimeError("memory unavailable"),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.exec_skill",
            new=AsyncMock(return_value=_skill_reply()),
        ) as skill:
            result = await run_crm_workflow(payload)
        return result, retrieve, skill

    result, retrieve, skill = asyncio.run(scenario())
    assert retrieve.await_count == expected_calls
    params = skill.await_args.args[1]
    if expected_calls:
        assert "[S1]" in params["knowledge_context"]
        assert params["citations"][0]["source_id"] == "policy-1"
        assert result.data["citations"][0]["citation_id"] == "S1"


def test_upstream_policy_context_prevents_retrieval():
    async def scenario():
        retrieve = AsyncMock()
        task = {
            "query": "帮我回复退款客户",
            "use_rag": True,
            "_upstream_context": {
                "policy_context": {
                    "task_type": "knowledge_qa",
                    "data": {
                        "answer": "Verified refund policy [S1]",
                        "citations": [
                            {"citation_id": "S1", "source_id": "policy-1"}
                        ],
                    },
                }
            },
        }
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.rag_service.retrieve",
            new=retrieve,
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.AgentShortMemory",
            side_effect=RuntimeError("memory unavailable"),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.exec_skill",
            new=AsyncMock(return_value=_skill_reply()),
        ) as skill:
            result = await run_crm_workflow(task)
        return result, retrieve, skill

    result, retrieve, skill = asyncio.run(scenario())
    retrieve.assert_not_awaited()
    assert "Verified refund policy" in skill.await_args.args[1]["knowledge_context"]
    assert result.data["rag_used"] is True


def test_retrieval_failure_is_safe_partial_success():
    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.rag_service.retrieve",
            new=AsyncMock(return_value=_retrieval(success=False)),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.AgentShortMemory",
            side_effect=RuntimeError("memory unavailable"),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.exec_skill",
            new=AsyncMock(return_value=_skill_reply()),
        ):
            return await run_crm_workflow({"query": "hello", "use_rag": True})

    result = asyncio.run(scenario())
    assert result.success and result.partial_success
    assert "RETRIEVAL_ERROR" in result.error_msg
    assert result.data["rag_used"] is False


def test_rag_agent_is_service_adapter_and_preserves_legacy_fields():
    async def scenario():
        queue: asyncio.Queue = asyncio.Queue()
        request = MCPMessage(
            task_id="root-rag",
            sender=AGENT_MASTER,
            target=AGENT_RAG,
            content={"query": "refund policy"},
        )
        document = _retrieval().documents[0]
        citation = _retrieval().citations[0]
        answer = RAGAnswerResult(
            success=True,
            answer="Refunds are accepted [S1].",
            documents=[document],
            citations=[citation],
            grounded=True,
            answer_source="test",
            cached=False,
            retrieval_latency_ms=1,
            total_latency_ms=2,
        )
        sent: list[MCPMessage] = []
        done = asyncio.Event()

        async def send(message):
            sent.append(message)
            done.set()
            return True

        await queue.put(request)
        with patch(
            "ecom_agent_matrix.modules.rag.rag_agent.rag_service.answer",
            new=AsyncMock(return_value=answer),
        ) as service, patch(
            "ecom_agent_matrix.modules.rag.rag_agent.mcp_bus.send_msg",
            new=AsyncMock(side_effect=send),
        ):
            task = asyncio.create_task(rag_agent(queue))
            await asyncio.wait_for(done.wait(), timeout=1)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return sent[0], service

    reply, service = asyncio.run(scenario())
    service.assert_awaited_once()
    data = reply.content["data"]
    for field in (
        "query", "lang", "recall_count", "latency_ms", "cached", "docs",
        "answer", "answer_source", "llm_error", "citations", "grounded",
        "retrieval_version",
    ):
        assert field in data
