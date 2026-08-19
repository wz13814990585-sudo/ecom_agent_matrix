"""异步消息总线、限流、重试、降级。"""
from __future__ import annotations

import asyncio
import json
from typing import Dict, List

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.db.base import AsyncPGClient
from ecom_agent_matrix.core.security import tenant_scope_from_security

logger = setup_logger("mcp.bus")


class MCPMessageBus:
    def __init__(self):
        # 审计/背压缓冲：记录近期消息量；分发以 agent_subscribe 为准
        self.queue_max = settings.MCP_QUEUE_MAX_SIZE
        self.msg_queue: asyncio.Queue[MCPMessage] = asyncio.Queue(maxsize=self.queue_max)
        self.agent_subscribe: Dict[str, List[asyncio.Queue]] = {}
        self.retry_times = settings.MCP_RETRY_TIMES

    async def _persist_msg(self, msg: MCPMessage) -> None:
        scope = tenant_scope_from_security(msg.security)
        if scope.usable:
            insert_sql = """
            INSERT INTO mcp_message_log(
              tenant_id, store_id, task_id, sender_agent, target_agent, priority, msg_content
            ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            """
            params = [
                scope.tenant_id, scope.store_id, msg.task_id, msg.sender,
                msg.target, msg.priority, json.dumps(msg.content, ensure_ascii=False),
            ]
        else:
            insert_sql = """
            INSERT INTO mcp_message_log(
              task_id, sender_agent, target_agent, priority, msg_content
            ) VALUES (%s, %s, %s, %s, %s::jsonb)
            """
            params = [
                msg.task_id, msg.sender, msg.target, msg.priority,
                json.dumps(msg.content, ensure_ascii=False),
            ]
        await AsyncPGClient.execute_write(
            insert_sql,
            params,
            scope=scope,
        )

    async def _put_audit_buffer(self, msg: MCPMessage) -> None:
        """写入全局缓冲；满则丢弃更低优先级消息腾出空间。"""
        if self.msg_queue.qsize() >= self.queue_max:
            kept: list[MCPMessage] = []
            while not self.msg_queue.empty():
                item = self.msg_queue.get_nowait()
                if item.priority <= msg.priority:
                    kept.append(item)
            for high_msg in kept:
                await self.msg_queue.put(high_msg)
            logger.warning(
                "mcp_queue_overload",
                extra={
                    "event": "mcp_queue_overload",
                    "kept": len(kept),
                    "task_id": msg.task_id,
                },
            )
        try:
            self.msg_queue.put_nowait(msg)
        except asyncio.QueueFull:
            # 极端情况：仍满则丢弃本条审计缓冲写入，不影响订阅分发
            logger.warning(
                "mcp_audit_drop",
                extra={"event": "mcp_audit_drop", "task_id": msg.task_id},
            )

    async def _dispatch_to_subscribers(self, msg: MCPMessage) -> bool:
        """推送到目标 Agent 订阅队列；无订阅者返回 False。"""
        target = msg.target
        queues = self.agent_subscribe.get(target) or []
        if not queues:
            return False
        for sub_queue in queues:
            await sub_queue.put(msg)
        return True

    async def send_msg(self, msg: MCPMessage) -> bool:
        """
        统一发送：审计缓冲 + DB 持久化 + 分发给订阅 Agent。
        若目标 Agent 尚未注册，按 MCP_RETRY_TIMES 短暂重试后再放弃。
        返回是否成功投递到至少一个订阅队列。
        """
        await self._put_audit_buffer(msg)

        try:
            await self._persist_msg(msg)
        except Exception as exc:
            # 持久化失败不阻断实时分发（本地/测试可能无表）
            logger.warning(
                "mcp_persist_failed",
                extra={
                    "event": "mcp_persist_failed",
                    "task_id": msg.task_id,
                    "error_type": type(exc).__name__,
                },
            )

        attempts = max(1, int(self.retry_times) + 1)
        delivered = False
        for attempt in range(attempts):
            if await self._dispatch_to_subscribers(msg):
                delivered = True
                if attempt > 0:
                    logger.info(
                        "mcp_dispatch_retry_ok",
                        extra={
                            "event": "mcp_dispatch_retry_ok",
                            "task_id": msg.task_id,
                            "target": msg.target,
                            "attempt": attempt + 1,
                        },
                    )
                break
            if attempt + 1 < attempts:
                await asyncio.sleep(0.05 * (attempt + 1))

        # HTTP Gateway 等待最终回传（不依赖 api_gateway 订阅队列）
        try:
            from ecom_agent_matrix.core.mcp.result_waiter import GatewayResultWaiter

            if GatewayResultWaiter.submit(msg):
                delivered = True
        except Exception as exc:
            logger.warning(
                "gateway_submit_failed",
                extra={"event": "gateway_submit_failed", "task_id": msg.task_id, "error_type": type(exc).__name__},
            )

        if not delivered:
            logger.error(
                "mcp_no_subscriber",
                extra={
                    "event": "mcp_no_subscriber",
                    "task_id": msg.task_id,
                    "target": msg.target,
                    "sender": msg.sender,
                    "retries": attempts,
                },
            )
        return delivered

    def register_agent(self, agent_id: str) -> asyncio.Queue:
        """Agent 注册：返回专属消费队列。"""
        agent_queue: asyncio.Queue = asyncio.Queue()
        if agent_id not in self.agent_subscribe:
            self.agent_subscribe[agent_id] = []
        self.agent_subscribe[agent_id].append(agent_queue)
        return agent_queue


mcp_bus = MCPMessageBus()
