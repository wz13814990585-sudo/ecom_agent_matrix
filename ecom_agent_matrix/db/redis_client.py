"""Redis 缓存/队列客户端。"""
import redis.asyncio as redis

from ecom_agent_matrix.config.settings import settings


class AsyncRedisClient:
    _client = None

    @classmethod
    async def get_client(cls):
        if cls._client is None:
            kwargs = {
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
                "db": settings.REDIS_DB,
                "decode_responses": True,
                "socket_timeout": float(settings.REDIS_SOCKET_TIMEOUT_SECONDS),
                "socket_connect_timeout": float(settings.REDIS_SOCKET_TIMEOUT_SECONDS),
            }
            pwd = (settings.REDIS_PASSWORD or "").strip()
            if pwd:
                kwargs["password"] = pwd
            cls._client = redis.Redis(**kwargs)
        return cls._client

    @classmethod
    async def close(cls):
        if cls._client:
            await cls._client.close()
            cls._client = None
