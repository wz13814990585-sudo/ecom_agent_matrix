"""限流与 CRM 辅助逻辑单测。"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.core.rate_limit import acquire_slot
from ecom_agent_matrix.modules.skills.crm_reply import format_rag_docs, should_use_rag
from ecom_agent_matrix.modules.skills.sql_tool import sanitize_readonly_sql


def test_should_use_rag_heuristic():
    assert should_use_rag("这款背包材质怎么样", None) is True
    assert should_use_rag("退款", True) is True
    assert should_use_rag("材质介绍", False) is False
    assert should_use_rag("我要退款", None) is False


def test_format_rag_docs():
    text = format_rag_docs(
        [{"title": "Bag", "content": "waterproof nylon"}, {"sku": "S1", "text": "light"}]
    )
    assert "Bag" in text and "waterproof" in text


def test_sanitize_readonly_sql():
    ok, err = sanitize_readonly_sql("SELECT sku FROM ecom_goods LIMIT 1")
    assert err == "" and ok and ok.lower().startswith("select")

    ok2, err2 = sanitize_readonly_sql("WITH t AS (SELECT 1) SELECT * FROM t")
    assert err2 == "" and ok2

    _, bad = sanitize_readonly_sql("SELECT 1; DELETE FROM ecom_goods")
    assert "多语句" in bad

    _, bad2 = sanitize_readonly_sql("UPDATE ecom_goods SET price=1")
    assert "仅允许" in bad2

    _, bad3 = sanitize_readonly_sql("SELECT * FROM ecom_goods; DROP TABLE ecom_goods")
    assert bad3


async def test_acquire_slot_process_mode():
    backends = []

    async def worker():
        async with acquire_slot("ut_social", limit=2, mode="process", ttl_sec=5) as b:
            backends.append(b)
            await asyncio.sleep(0.01)

    await asyncio.gather(*(worker() for _ in range(3)))
    assert backends == ["process", "process", "process"]


async def test_acquire_slot_redis_falls_back():
    with patch(
        "ecom_agent_matrix.core.rate_limit.AsyncRedisClient.get_client",
        new=AsyncMock(side_effect=RuntimeError("redis down")),
    ):
        async with acquire_slot("ut_social_fb", limit=1, mode="redis", ttl_sec=2) as b:
            assert b == "process_fallback"


if __name__ == "__main__":
    test_should_use_rag_heuristic()
    test_format_rag_docs()
    test_sanitize_readonly_sql()
    asyncio.run(test_acquire_slot_process_mode())
    asyncio.run(test_acquire_slot_redis_falls_back())
    print("✅ rate_limit / crm helpers / sql sanitize ok")
