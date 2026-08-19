"""FastAPI 入口：HTTP 与 Agent 矩阵同进程启动。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Response, status

# 侧载注册 Agent / Skill
import ecom_agent_matrix.modules.agent_cluster  # noqa: F401
import ecom_agent_matrix.modules.skills  # noqa: F401
from ecom_agent_matrix.api.auth import require_api_key
from ecom_agent_matrix.api.health import readiness_report
from ecom_agent_matrix.api.route_customer import router as customer_router
from ecom_agent_matrix.api.route_task import router as task_router
from ecom_agent_matrix.api.route_warn import router as warn_router
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.llm.deepseek_client import close_http_session
from ecom_agent_matrix.core.mcp.registry import agent_map, start_all_agents
from ecom_agent_matrix.core.skill.skill_registry import skill_container

logger = setup_logger("api.main")

_agent_task: asyncio.Task | None = None

_OPENAPI_TAGS = [
    {"name": "system", "description": "健康检查与注册表"},
    {"name": "tasks", "description": "经 Master ReAct 的通用运营任务"},
    {"name": "customer", "description": "店铺规则 / 售后问答（经 Master → RAG 或查询）"},
    {"name": "warn", "description": "竞品价格查询（经 Master 或直达 Query）"},
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _agent_task
    api_key_on = bool((settings.API_KEY or "").strip())
    if not api_key_on:
        logger.warning(
            "api_auth_disabled",
            extra={
                "event": "api_auth_disabled",
                "error": "API_KEY 为空，HTTP 接口未鉴权（仅适合本地开发）",
            },
        )
    logger.info(
        "api_starting",
        extra={
            "event": "api_starting",
            "agents": sorted(agent_map.keys()),
            "api_auth_enabled": api_key_on,
        },
    )
    if not agent_map:
        raise RuntimeError("agent_map 为空：侧载注册失败")
    _agent_task = asyncio.create_task(start_all_agents(), name="start_all_agents")
    await asyncio.sleep(0.05)
    yield
    if _agent_task and not _agent_task.done():
        _agent_task.cancel()
        try:
            await _agent_task
        except asyncio.CancelledError:
            pass
    await close_http_session()


app = FastAPI(
    title="Ecom Agent Matrix",
    version="0.1.0",
    description=(
        "跨境独立站电商多智能体矩阵 HTTP 网关。\n\n"
        "- 交互文档：`/docs`（Swagger）或 `/redoc`\n"
        "- 鉴权：配置 `API_KEY` 后请求头携带 `X-API-Key`\n"
        "- 通用任务走 Master（规划/分发/聚合）；子 Agent 仅 Query / Exec / RAG"
    ),
    lifespan=lifespan,
    openapi_tags=_OPENAPI_TAGS,
)
app.include_router(task_router)
app.include_router(customer_router)
app.include_router(warn_router)


@app.get("/health", tags=["system"], summary="存活探针（不探测外部依赖）")
async def health():
    return {
        "status": "ok",
        "agents_running": bool(_agent_task and not _agent_task.done()),
        "agents_count": len(agent_map),
        "skills_count": len(skill_container),
        "api_auth_enabled": bool((settings.API_KEY or "").strip()),
    }


@app.get("/health/ready", tags=["system"], summary="就绪探针（Postgres + Redis）")
async def health_ready(response: Response):
    report = await readiness_report()
    if not report["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if report["ready"] else "not_ready",
        "agents_running": bool(_agent_task and not _agent_task.done()),
        **report,
    }


@app.get(
    "/api/v1/agents",
    tags=["system"],
    summary="已注册 Agent / Skill",
    dependencies=[Depends(require_api_key)],
)
async def list_agents():
    return {"agents": sorted(agent_map.keys()), "skills": sorted(skill_container.keys())}
