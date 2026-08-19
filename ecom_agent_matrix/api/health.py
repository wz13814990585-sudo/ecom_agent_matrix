"""依赖健康探测（Postgres / Redis）。"""
from __future__ import annotations

from typing import Any

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger

logger = setup_logger("api.health")


async def check_postgres() -> dict[str, Any]:
    try:
        from ecom_agent_matrix.db.base import AsyncPGClient

        read_rows = await AsyncPGClient.execute_health("read")
        write_rows = await AsyncPGClient.execute_health("write")
        ok = bool(
            read_rows and read_rows[0] and read_rows[0][0] == 1
            and write_rows and write_rows[0] and write_rows[0][0] == 1
        )
        return {"ok": ok, "status": "ok" if ok else "degraded"}
    except Exception as exc:
        logger.warning(
            "health_pg_failed",
            extra={"event": "health_pg_failed", "error_type": type(exc).__name__},
        )
        return {"ok": False, "status": "degraded", "error_code": "DEPENDENCY_UNAVAILABLE"}


async def check_redis() -> dict[str, Any]:
    try:
        from ecom_agent_matrix.db.redis_client import AsyncRedisClient

        client = await AsyncRedisClient.get_client()
        pong = await client.ping()
        return {"ok": bool(pong), "status": "ok" if pong else "degraded"}
    except Exception as exc:
        logger.warning(
            "health_redis_failed",
            extra={"event": "health_redis_failed", "error_type": type(exc).__name__},
        )
        return {"ok": False, "status": "degraded", "error_code": "DEPENDENCY_UNAVAILABLE"}


async def readiness_report(*, agents_alive: bool = True) -> dict[str, Any]:
    from ecom_agent_matrix.core.llm.router import is_llm_configured

    pg = await check_postgres()
    redis = await check_redis()
    llm_status = "unknown" if is_llm_configured() else "degraded"
    llm_ok = llm_status != "degraded" or not bool(settings.LLM_REQUIRED_FOR_READINESS)
    ready = bool(pg.get("ok") and redis.get("ok") and agents_alive and llm_ok)
    return {
        "ready": ready,
        "degraded": llm_status == "degraded",
        "dependencies": {
            "postgres": pg["status"],
            "redis": redis["status"],
            "llm": llm_status,
            "agents": "ok" if agents_alive else "degraded",
        },
    }
