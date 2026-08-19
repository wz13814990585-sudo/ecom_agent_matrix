"""短记忆：滑动窗口 + 并发 append 不丢消息。"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.core.memory.short_memory import AgentShortMemory
from ecom_agent_matrix.db.redis_client import AsyncRedisClient


async def test_sliding_window():
    sid = f"test-win-{uuid.uuid4().hex[:8]}"
    mem = AgentShortMemory(session_id=sid, ttl=60, max_messages=4)
    await mem.clear()

    for i in range(6):
        await mem.append("user" if i % 2 == 0 else "assistant", f"msg-{i}")

    msgs = await mem.get_all()
    print("=== 滑动窗口（max=4，写入 6 条）===")
    print("条数:", len(msgs))
    for m in msgs:
        print(f"  {m['role']}: {m['content']}")
    assert len(msgs) == 4
    assert [m["content"] for m in msgs] == ["msg-2", "msg-3", "msg-4", "msg-5"]
    await mem.clear()
    print("滑动窗口 OK\n")


async def test_concurrent_append():
    sid = f"test-conc-{uuid.uuid4().hex[:8]}"
    mem = AgentShortMemory(session_id=sid, ttl=60, max_messages=100)
    await mem.clear()

    n = 40

    async def writer(i: int):
        await mem.append("user", f"c-{i}")

    await asyncio.gather(*[writer(i) for i in range(n)])
    msgs = await mem.get_all()
    contents = sorted(m["content"] for m in msgs)
    expected = sorted(f"c-{i}" for i in range(n))
    print("=== 并发 append ===")
    print("期望条数:", n, "实际:", len(msgs))
    print("前 5 条内容:", [m["content"] for m in msgs[:5]], "...")
    assert len(msgs) == n, f"丢失消息：got {len(msgs)} expected {n}"
    assert contents == expected
    await mem.clear()
    print("并发 append OK\n")


async def test_legacy_migrate():
    """旧 string JSON → list 迁移。"""
    sid = f"test-legacy-{uuid.uuid4().hex[:8]}"
    key = f"agent:short_mem:{sid}"
    redis = await AsyncRedisClient.get_client()
    await redis.delete(key)
    await redis.set(
        key,
        '[{"role":"user","content":"old-1"},{"role":"assistant","content":"old-2"}]',
        ex=60,
    )

    mem = AgentShortMemory(session_id=sid, ttl=60, max_messages=20)
    msgs = await mem.get_all()
    print("=== 旧格式迁移 ===")
    print(msgs)
    assert len(msgs) == 2
    assert msgs[0]["content"] == "old-1"

    key_type = await redis.type(key)
    print("迁移后类型:", key_type)
    assert key_type in ("list", b"list")

    await mem.append("user", "new-3")
    msgs2 = await mem.get_all()
    print("再 append 后:", msgs2)
    assert len(msgs2) == 3
    await mem.clear()
    print("旧格式迁移 OK\n")


async def main():
    try:
        await test_sliding_window()
        await test_concurrent_append()
        await test_legacy_migrate()
        print("短记忆改造验证完成")
    finally:
        await AsyncRedisClient.close()


if __name__ == "__main__":
    asyncio.run(main())
