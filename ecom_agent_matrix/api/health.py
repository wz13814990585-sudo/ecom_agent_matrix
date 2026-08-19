"""依赖健康探测（Postgres / Redis）。"""
from __future__ import annotations

from typing import Any

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger

logger = setup_logger("api.health")


async def check_postgres() -> dict[str, Any]:
    try:
        from ecom_agent_matrix.db.base import AsyncPGClient

        rows = await AsyncPGClient.execute_sql("SELECT 1 AS ok")
        ok = bool(rows and rows[0] and rows[0][0] == 1)
        return {
            "ok": ok,
            "host": settings.PG_HOST,
            "port": settings.PG_PORT,
            "db": settings.PG_DB,
        }
    except Exception as exc:
        logger.warning(
            "health_pg_failed",
            extra={"event": "health_pg_failed", "error": str(exc)},
        )
        return {
            "ok": False,
            "host": settings.PG_HOST,
            "port": settings.PG_PORT,
            "db": settings.PG_DB,
            "error": str(exc),
        }


async def check_redis() -> dict[str, Any]:
    try:
        from ecom_agent_matrix.db.redis_client import AsyncRedisClient

        client = await AsyncRedisClient.get_client()
        pong = await client.ping()
        return {
            "ok": bool(pong),
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "db": settings.REDIS_DB,
        }
    except Exception as exc:
        logger.warning(
            "health_redis_failed",
            extra={"event": "health_redis_failed", "error": str(exc)},
        )
        return {
            "ok": False,
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "db": settings.REDIS_DB,
            "error": str(exc),
        }


async def readiness_report() -> dict[str, Any]:
    pg = await check_postgres()
    redis = await check_redis()
    ready = bool(pg.get("ok") and redis.get("ok"))
    return {
        "ready": ready,
        "postgres": pg,
        "redis": redis,
    }
