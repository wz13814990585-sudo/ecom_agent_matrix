"""Agent 短记忆：Redis List + 滑动窗口，并发 append 安全。"""
from __future__ import annotations

import json
import hashlib
from typing import Any

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.db.redis_client import AsyncRedisClient

# RPUSH + LTRIM + EXPIRE 原子执行，避免 get→set 并发覆盖
_APPEND_LUA = """
redis.call('RPUSH', KEYS[1], ARGV[1])
local max_n = tonumber(ARGV[2])
if max_n and max_n > 0 then
  redis.call('LTRIM', KEYS[1], -max_n, -1)
end
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
return redis.call('LLEN', KEYS[1])
"""


class AgentShortMemory:
    """
    短期会话记忆（CRM 等）。
    - 存储：Redis List（每条消息一个 JSON 元素）
    - 并发：Lua 脚本原子 append，不会互相覆盖
    - 窗口：只保留最近 max_messages 条
    """

    def __init__(
        self,
        session_id: str,
        ttl: int | None = None,
        max_messages: int | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ):
        self.session_id = session_id
        self.ttl = int(ttl if ttl is not None else settings.SHORT_MEMORY_TTL)
        self.max_messages = int(
            max_messages
            if max_messages is not None
            else settings.SHORT_MEMORY_MAX_MESSAGES
        )
        # list: 新结构；旧 string JSON 会在首次读写时迁移
        if tenant_id and user_id:
            tenant_key = hashlib.sha256(str(tenant_id).encode("utf-8")).hexdigest()[:24]
            user_key = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:24]
            session_key = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:24]
            self.key = f"agent:short_mem:{tenant_key}:{user_key}:{session_key}"
        else:
            self.key = f"agent:short_mem:{session_id}"

    async def append(self, role: str, content: str) -> int:
        """追加一条消息并滑动截断，返回当前列表长度。"""
        redis = await AsyncRedisClient.get_client()
        await self._migrate_legacy_if_needed(redis)

        item = json.dumps(
            {"role": str(role), "content": str(content)},
            ensure_ascii=False,
        )
        length = await redis.eval(
            _APPEND_LUA,
            1,
            self.key,
            item,
            self.max_messages,
            self.ttl,
        )
        return int(length or 0)

    async def get_all(self) -> list[dict[str, Any]]:
        redis = await AsyncRedisClient.get_client()
        await self._migrate_legacy_if_needed(redis)

        raw_list = await redis.lrange(self.key, 0, -1)
        msgs: list[dict[str, Any]] = []
        for raw in raw_list or []:
            try:
                obj = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(obj, dict):
                msgs.append(obj)
        return msgs

    async def clear(self) -> None:
        redis = await AsyncRedisClient.get_client()
        await redis.delete(self.key)

    async def _migrate_legacy_if_needed(self, redis) -> None:
        """旧版把整个 msgs 存成一个 JSON string；迁移为 list，避免类型冲突。"""
        key_type = await redis.type(self.key)
        if key_type in (None, "none", b"none"):
            return
        if key_type in ("list", b"list"):
            return
        if key_type not in ("string", b"string"):
            # 非预期类型：清掉重建
            await redis.delete(self.key)
            return

        raw = await redis.get(self.key)
        try:
            msgs = json.loads(raw) if raw else []
        except (TypeError, json.JSONDecodeError):
            msgs = []
        if not isinstance(msgs, list):
            msgs = []

        pipe = redis.pipeline()
        pipe.delete(self.key)
        if msgs:
            payload = [
                json.dumps(m, ensure_ascii=False)
                for m in msgs
                if isinstance(m, dict)
            ]
            if payload:
                # 先截断再写入，避免一次灌入超长历史
                keep = payload[-self.max_messages :] if self.max_messages > 0 else payload
                pipe.rpush(self.key, *keep)
                pipe.expire(self.key, self.ttl)
        await pipe.execute()
