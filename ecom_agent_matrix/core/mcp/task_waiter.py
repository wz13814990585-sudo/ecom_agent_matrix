"""MCP 子任务回传等待器：Master 等待子 Agent 结果。"""
from __future__ import annotations

import asyncio
from typing import Optional

from ecom_agent_matrix.core.mcp.message import MCPMessage

REPLY_TYPES = frozenset({"agent_reply", "rag_retrieve_result", "sub_agent_reply"})


def is_agent_reply(msg: MCPMessage) -> bool:
    """判断是否为子 Agent 回传消息。"""
    return msg.content.get("type") in REPLY_TYPES


class TaskReplyWaiter:
    """按 task_id 收集子 Agent 回传，支持超时。"""

    _pending: dict[str, dict] = {}

    @classmethod
    def begin(cls, task_id: str, expected_count: int) -> None:
        cls._pending[task_id] = {
            "expected": expected_count,
            "replies": [],
            "event": asyncio.Event(),
        }

    @classmethod
    def submit_reply(cls, msg: MCPMessage) -> bool:
        ref_id = msg.content.get("ref_task_id") or msg.task_id
        buf = cls._pending.get(ref_id)
        if not buf:
            return False
        buf["replies"].append(msg)
        if len(buf["replies"]) >= buf["expected"]:
            buf["event"].set()
        return True

    @classmethod
    async def wait(cls, task_id: str, timeout: float) -> list[MCPMessage]:
        buf = cls._pending.get(task_id)
        if not buf:
            return []
        try:
            await asyncio.wait_for(buf["event"].wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        replies = list(buf["replies"])
        cls._pending.pop(task_id, None)
        return replies

    @classmethod
    def pending_count(cls, task_id: str) -> int:
        buf = cls._pending.get(task_id)
        if not buf:
            return 0
        return buf["expected"] - len(buf["replies"])
