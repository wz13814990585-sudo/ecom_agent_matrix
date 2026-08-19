# test_message.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.config.constants import AGENT_MASTER, AGENT_RAG, MSG_PRIORITY_CUSTOMER

async def test_msg_model():
    # 模拟总控Agent下发商品检索任务给RAG知识库Agent
    msg = MCPMessage(
        sender=AGENT_MASTER,
        target=AGENT_RAG,
        priority=MSG_PRIORITY_CUSTOMER,
        content={"query": "平价海边连衣裙", "lang": "en", "price_max": 80}
    )
    print("消息完整结构体：")
    print(msg.model_dump_json(indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_msg_model())