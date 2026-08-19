# test_bus.py
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.core.mcp.bus import mcp_bus
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.config.constants import (
    AGENT_MASTER,
    AGENT_RAG,
    MSG_PRIORITY_CUSTOMER,
    MSG_PRIORITY_NORMAL,
    MSG_PRIORITY_AD,
    MSG_PRIORITY_SOCIAL,
)

# 批量测试消息（会写入 mcp_message_log，并触发消费者打印）
TEST_TASKS = [
    {"query": "waterproof outdoor bag", "lang": "en", "priority": MSG_PRIORITY_CUSTOMER},
    {"query": "平价海边连衣裙", "lang": "zh", "priority": MSG_PRIORITY_CUSTOMER},
    {"query": "bolso de playa barato", "lang": "es", "priority": MSG_PRIORITY_NORMAL},
    {"query": "sac de voyage étanche", "lang": "fr", "priority": MSG_PRIORITY_AD},
    {"query": "summer festival outfit", "lang": "en", "priority": MSG_PRIORITY_SOCIAL},
]


async def consumer_rag(queue: asyncio.Queue):
    # 模拟RAG Agent持续消费消息
    while True:
        msg = await queue.get()
        print(f"【RAG Agent收到任务】task_id:{msg.task_id} 参数:{msg.content}")
        queue.task_done()


async def main():
    # 1.RAG Agent注册订阅总线
    rag_queue = mcp_bus.register_agent(AGENT_RAG)
    # 启动RAG消费协程
    asyncio.create_task(consumer_rag(rag_queue))

    # 2.总控Agent批量发送检索消息
    for task in TEST_TASKS:
        test_msg = MCPMessage(
            sender=AGENT_MASTER,
            target=AGENT_RAG,
            priority=task["priority"],
            content={"query": task["query"], "lang": task["lang"]},
        )
        await mcp_bus.send_msg(test_msg)

    await asyncio.sleep(1)
    print(f"✅ 共发送并消费 {len(TEST_TASKS)} 条消息")


if __name__ == "__main__":
    asyncio.run(main())
