import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.db.redis_client import AsyncRedisClient

async def test_redis():
    r = await AsyncRedisClient.get_client()
    # 写入缓存：模拟商品向量缓存
    await r.set("goods_sku_001_embedding", "test_vector_data", ex=3600)
    val = await r.get("goods_sku_001_embedding")
    print("Redis缓存读写成功：", val)

if __name__ == "__main__":
    asyncio.run(test_redis())