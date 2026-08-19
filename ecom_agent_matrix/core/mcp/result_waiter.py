"""HTTP API 等待最终 Agent 回传（与 Master 子任务 TaskReplyWaiter 隔离）。"""
from __future__ import annotations

import asyncio

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.mcp.message import MCPMessage

# Master 最终结果 + 常规子 Agent 回传
GATEWAY_REPLY_TYPES = frozenset(
    {"agent_reply", "rag_retrieve_result", "sub_agent_reply", "master_task_result"}
)


class GatewayResultWaiter:
    """按 task_id 等待一条发给 API 的最终回传。"""

    _pending: dict[str, dict] = {}

    @classmethod
    def begin(cls, task_id: str) -> None:
        cls._pending[task_id] = {
            "reply": None,
            "event": asyncio.Event(),
        }

    @classmethod
    def submit(cls, msg: MCPMessage) -> bool:
        # 只收发给 api_gateway 的回传，避免 Master 子 Agent 回传误触等待器
        if msg.target != settings.API_SENDER:
            return False
        if msg.content.get("type") not in GATEWAY_REPLY_TYPES:
            return False
        ref_id = str(msg.content.get("ref_task_id") or msg.task_id)
        buf = cls._pending.get(ref_id)
        if not buf:
            return False
        buf["reply"] = msg
        buf["event"].set()
        return True

    @classmethod
    async def wait(cls, task_id: str, timeout: float) -> MCPMessage | None:
        buf = cls._pending.get(task_id)
        if not buf:
            return None
        try:
            await asyncio.wait_for(buf["event"].wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        reply = buf.get("reply")
        cls._pending.pop(task_id, None)
        return reply

    @classmethod
    def cancel(cls, task_id: str) -> None:
        cls._pending.pop(task_id, None)
