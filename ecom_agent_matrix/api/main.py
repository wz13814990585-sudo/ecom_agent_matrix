"""FastAPI 入口：HTTP 与 Agent 矩阵同进程启动。"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import PlainTextResponse

# 侧载注册 Agent / Skill
import ecom_agent_matrix.modules.agent_cluster  # noqa: F401
import ecom_agent_matrix.modules.skills  # noqa: F401
from ecom_agent_matrix.api.auth import (
    get_current_security_context,
    validate_security_configuration,
)
from ecom_agent_matrix.api.health import readiness_report
from ecom_agent_matrix.api.route_customer import router as customer_router
from ecom_agent_matrix.api.route_task import router as task_router
from ecom_agent_matrix.api.route_warn import router as warn_router
from ecom_agent_matrix.api.route_approval import router as approval_router
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.llm import close_http_session
from ecom_agent_matrix.core.mcp.registry import agent_map, start_all_agents
from ecom_agent_matrix.core.skill.skill_registry import skill_container
from ecom_agent_matrix.modules.agent_cluster.master_agent import cancel_master_tasks
from ecom_agent_matrix.db.base import (
    AsyncPGClient,
    validate_database_runtime_roles,
    validate_database_security_configuration,
)
from ecom_agent_matrix.db.redis_client import AsyncRedisClient
from ecom_agent_matrix.platform.observability.context import (
    TraceContext,
    begin_request_performance,
    finish_request_performance,
    get_trace_context,
    trace_context,
)
from ecom_agent_matrix.platform.observability.metrics import metrics
from ecom_agent_matrix.core.security import effective_scopes

logger = setup_logger("api.main")

_agent_task: asyncio.Task | None = None
_accepting_requests = True

_OPENAPI_TAGS = [
    {"name": "system", "description": "健康检查与注册表"},
    {"name": "tasks", "description": "经 Master Fast Path / Typed DAG / Recovery 的通用运营任务"},
    {"name": "customer", "description": "店铺规则 / 售后问答（经 Master → RAG 或查询）"},
    {"name": "warn", "description": "竞品价格查询（经 Master 或直达 Query）"},
    {"name": "approvals", "description": "高风险写操作人工审批"},
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _agent_task, _accepting_requests
    _accepting_requests = True
    validate_security_configuration(settings)
    validate_database_security_configuration(settings)
    await validate_database_runtime_roles(settings)
    logger.info(
        "api_starting",
        extra={
            "event": "api_starting",
            "agents": sorted(agent_map.keys()),
            "auth_mode": settings.AUTH_MODE,
        },
    )
    if not agent_map:
        raise RuntimeError("agent_map 为空：侧载注册失败")
    _agent_task = asyncio.create_task(start_all_agents(), name="start_all_agents")
    await asyncio.sleep(0.05)
    yield
    _accepting_requests = False
    await shutdown_runtime(_agent_task)


async def shutdown_runtime(agent_task: asyncio.Task | None) -> None:
    """Bounded cleanup owned by the FastAPI lifespan."""
    async def cleanup():
        if agent_task and not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
        await cancel_master_tasks()
        await close_http_session()
        await AsyncRedisClient.close()
        await AsyncPGClient.close()

    try:
        await asyncio.wait_for(cleanup(), timeout=float(settings.SHUTDOWN_TIMEOUT_SECONDS))
    except asyncio.TimeoutError:
        logger.error(
            "shutdown_timeout",
            extra={"event": "shutdown_timeout", "error_code": "TIMEOUT", "component": "runtime"},
        )


app = FastAPI(
    title="Ecom Agent Matrix",
    version="0.1.0",
    description=(
        "跨境独立站电商多智能体矩阵 HTTP 网关。\n\n"
        "- 交互文档：`/docs`（Swagger）或 `/redoc`\n"
        "- 鉴权：开发环境 X-API-Key，生产环境 JWT Bearer token\n"
        "- 通用任务走 Master（规划/分发/聚合）；子 Agent 仅 Query / Exec / RAG"
    ),
    lifespan=lifespan,
    openapi_tags=_OPENAPI_TAGS,
)
app.include_router(task_router)
app.include_router(customer_router)
app.include_router(warn_router)
app.include_router(approval_router)


@app.middleware("http")
async def observe_http_request(request: Request, call_next):
    started = time.perf_counter()
    status_code = 500
    task_id = str(uuid.uuid4())
    begin_request_performance(task_id)
    with trace_context(TraceContext(task_id=task_id, request_started_at=time.time())):
        try:
            if not _accepting_requests and request.url.path not in {"/health", "/health/ready"}:
                response = PlainTextResponse("DEPENDENCY_UNAVAILABLE", status_code=503)
            else:
                response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            route_obj = request.scope.get("route")
            route = getattr(route_obj, "path", None) or "unmatched"
            latency = time.perf_counter() - started
            metrics.observe_http(request.method, route, status_code, latency)
            trace = get_trace_context()
            logger.info(
                "http_request_completed",
                extra={
                    "event": "http_request_completed", "method": request.method,
                    "route": route, "status": status_code,
                    "latency_ms": round(latency * 1000, 2),
                    "task_id": trace.task_id, "correlation_id": trace.correlation_id,
                },
            )
            finish_request_performance(task_id)


@app.get("/health", tags=["system"], summary="存活探针（不探测外部依赖）")
async def health():
    return {
        "status": "ok",
        "agents_running": bool(_agent_task and not _agent_task.done()),
        "agents_count": len(agent_map),
        "skills_count": len(skill_container),
        "api_auth_enabled": True,
    }


@app.get("/health/ready", tags=["system"], summary="就绪探针（Postgres + Redis）")
async def health_ready(response: Response):
    agents_alive = bool(_agent_task and not _agent_task.done())
    report = await readiness_report(agents_alive=agents_alive)
    if not report["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if report["ready"] else "not_ready",
        "agents_running": agents_alive,
        **report,
    }


@app.get("/metrics", tags=["system"], include_in_schema=False)
async def prometheus_metrics(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if settings.METRICS_AUTH_REQUIRED or str(settings.APP_ENV).lower() == "production":
        security = await get_current_security_context(authorization, x_api_key)
        if "system:read" not in effective_scopes(security):
            raise HTTPException(status_code=403, detail="PERMISSION_DENIED")
    return Response(content=metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get(
    "/api/v1/agents",
    tags=["system"],
    summary="已注册 Agent / Skill",
    dependencies=[Depends(get_current_security_context)],
)
async def list_agents():
    return {"agents": sorted(agent_map.keys()), "skills": sorted(skill_container.keys())}
