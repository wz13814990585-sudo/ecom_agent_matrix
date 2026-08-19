"""RAG Agent 集成测试（mock 检索，不依赖向量库）。"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.config.constants import AGENT_MASTER, AGENT_RAG, MSG_PRIORITY_CUSTOMER
from ecom_agent_matrix.core.mcp.bus import mcp_bus
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.reply import build_rag_reply
from ecom_agent_matrix.modules.rag import rag_agent as rag_agent_module  # noqa: F401


async def test_rag_reply_task_id():
    sent = []

    async def fake_send(msg: MCPMessage):
        sent.append(msg)

    mock_docs = [{"sku": "SKU-001", "chunk_text": "短袖连衣裙", "relevance_score": 0.9}]
    req = MCPMessage(
        sender=AGENT_MASTER,
        target=AGENT_RAG,
        priority=MSG_PRIORITY_CUSTOMER,
        content={"query": "半袖", "lang": "zh", "top_k": 5},
    )

    with patch.object(mcp_bus, "send_msg", side_effect=fake_send), patch(
        "ecom_agent_matrix.modules.rag.retriever.hybrid_retrieve",
        new=AsyncMock(return_value=(mock_docs, True, 5.0)),
    ), patch(
        "ecom_agent_matrix.modules.rag.rag_agent.llm_explain",
        new=AsyncMock(return_value=("根据知识库，该款为短袖连衣裙。", "deepseek", "")),
    ):
        # 直接走一次 agent 队列逻辑较重，这里验证回传结构含 answer
        reply = build_rag_reply(
            req,
            query="半袖",
            lang="zh",
            docs=mock_docs,
            recall_count=1,
            latency_ms=20,
            cached=True,
            answer="根据知识库，该款为短袖连衣裙。",
            answer_source="deepseek",
        )
        await fake_send(reply)

    assert len(sent) == 1
    r = sent[0]
    assert r.task_id == req.task_id
    assert r.target == AGENT_MASTER
    assert r.content["ref_task_id"] == req.task_id
    assert r.content["data"]["cached"] is True
    assert r.content["data"]["answer"]
    assert r.content["data"]["answer_source"] == "deepseek"
    print("✅ MCP 回传 task_id 对齐测试通过")


async def test_rag_agent_generates_answer():
    sent = []

    async def fake_send(msg: MCPMessage):
        sent.append(msg)

    q: asyncio.Queue = asyncio.Queue()
    mock_docs = [{"sku": "SKU-001", "chunk_text": "防水户外背包，容量 30L", "relevance_score": 0.9}]
    req = MCPMessage(
        sender=AGENT_MASTER,
        target=AGENT_RAG,
        priority=MSG_PRIORITY_CUSTOMER,
        content={"query": "背包容量多少", "lang": "zh", "top_k": 3},
    )

    task = asyncio.create_task(rag_agent_module.rag_agent(q))
    try:
        with patch.object(mcp_bus, "send_msg", side_effect=fake_send), patch(
            "ecom_agent_matrix.modules.rag.rag_agent.hybrid_retrieve",
            new=AsyncMock(return_value=(mock_docs, False, 8.0)),
        ), patch(
            "ecom_agent_matrix.modules.rag.rag_agent.llm_explain",
            new=AsyncMock(return_value=("容量约 30L。", "deepseek", "")),
        ):
            await q.put(req)
            for _ in range(40):
                if sent:
                    break
                await asyncio.sleep(0.05)
        assert sent, "rag_agent 未回传"
        data = sent[0].content["data"]
        assert data["answer"] == "容量约 30L。"
        assert data["answer_source"] == "deepseek"
        assert data["recall_count"] == 1
        print("✅ RAG Agent LLM 答复测试通过")
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(test_rag_reply_task_id())
    asyncio.run(test_rag_agent_generates_answer())
