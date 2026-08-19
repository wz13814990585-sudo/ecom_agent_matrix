"""Master 共享路由 policy 与 DAG validation。"""
from __future__ import annotations

import re

from ecom_agent_matrix.config.constants import (
    AGENT_AD,
    AGENT_CRM,
    AGENT_DATA_CHECK,
    AGENT_EXEC,
    AGENT_GOODS,
    AGENT_PRICE_WARN,
    AGENT_QUERY,
    AGENT_RAG,
    AGENT_REPORT,
    AGENT_SOCIAL,
    AGENT_STOCK,
)
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.modules.agent_cluster.master.schemas import MasterPlan

AVAILABLE_AGENTS: dict[str, str] = {
    AGENT_QUERY: "只读数据查询：广告、订单、库存、竞品、商品目录。",
    AGENT_EXEC: "业务执行与产物：广告优化、风控、报表、社媒、客服回复。",
    AGENT_RAG: "店铺政策、运营手册、FAQ 与商品知识。",
}

TASK_ROUTE_MAP: dict[str, list[str]] = {
    "customer_service": [AGENT_EXEC],
    "stock_analysis": [AGENT_QUERY],
    "social_marketing": [AGENT_EXEC],
    "competitor_watch": [AGENT_QUERY],
    "goods_search": [AGENT_QUERY],
    "goods_catalog": [AGENT_QUERY],
    "knowledge_qa": [AGENT_RAG],
    "ad_optimize": [AGENT_EXEC],
    "ad_query": [AGENT_QUERY],
    "data_check": [AGENT_QUERY],
    "order_query": [AGENT_QUERY],
    "ops_report": [AGENT_EXEC],
    "risk_control": [AGENT_EXEC],
}
ALLOWED_TASK_TYPES = frozenset(TASK_ROUTE_MAP)

AGENT_ID_ALIASES: dict[str, str] = {
    AGENT_GOODS: AGENT_QUERY,
    AGENT_STOCK: AGENT_QUERY,
    AGENT_DATA_CHECK: AGENT_QUERY,
    AGENT_PRICE_WARN: AGENT_QUERY,
    "stock_predict": AGENT_QUERY,
    "data_integrity_check": AGENT_QUERY,
    "goods_lookup": AGENT_QUERY,
    "goods_rag": AGENT_RAG,
    AGENT_AD: AGENT_EXEC,
    AGENT_REPORT: AGENT_EXEC,
    AGENT_SOCIAL: AGENT_EXEC,
    AGENT_CRM: AGENT_EXEC,
    "ops_report": AGENT_EXEC,
    "ad_optimizer": AGENT_EXEC,
    "customer_service": AGENT_EXEC,
}

PLAN_CYCLE = "PLAN_CYCLE"
INVALID_DEPENDENCY = "INVALID_DEPENDENCY"
INVALID_AGENT = "INVALID_AGENT"
INVALID_TASK_TYPE = "INVALID_TASK_TYPE"
DUPLICATE_STEP_ID = "DUPLICATE_STEP_ID"
SELF_DEPENDENCY = "SELF_DEPENDENCY"
PLAN_TOO_LARGE = "PLAN_TOO_LARGE"
INVALID_AGENT_ROUTE = "INVALID_AGENT_ROUTE"

_ORDER_CONTEXT = re.compile(
    r"订单状态|物流|发货|tracking|order\s+status|ORD[-_][A-Z0-9_-]+", re.I
)
_POLICY_CONTEXT = re.compile(
    r"退款规则|退货政策|店铺规则|refund\s+policy|return\s+policy", re.I
)
_CUSTOMER_REPLY = re.compile(
    r"回复.*(?:客户|顾客|买家)|帮我回复|客服回复|customer\s+reply", re.I
)


def is_composite_customer_reply(query: str) -> bool:
    """订单事实 + 政策知识 + 客服回复的高置信策略判定。"""
    return bool(
        _ORDER_CONTEXT.search(query or "")
        and _POLICY_CONTEXT.search(query or "")
        and _CUSTOMER_REPLY.search(query or "")
    )


class MasterPlanValidationError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def validate_master_plan(plan: MasterPlan) -> MasterPlan:
    """严格验证 step identity、依赖引用、policy mapping 与 DAG 无环性。"""
    if len(plan.steps) > int(settings.MASTER_MAX_PLAN_STEPS):
        raise MasterPlanValidationError(PLAN_TOO_LARGE, "plan steps 超过上限")

    ids = [step.step_id for step in plan.steps]
    if len(ids) != len(set(ids)):
        raise MasterPlanValidationError(DUPLICATE_STEP_ID, "step_id 必须唯一")
    known = set(ids)
    for step in plan.steps:
        if step.agent not in AVAILABLE_AGENTS:
            raise MasterPlanValidationError(INVALID_AGENT, f"未知 Agent: {step.agent}")
        if step.task_type not in ALLOWED_TASK_TYPES:
            raise MasterPlanValidationError(
                INVALID_TASK_TYPE, f"未知 task_type: {step.task_type}"
            )
        if step.agent not in TASK_ROUTE_MAP[step.task_type]:
            raise MasterPlanValidationError(
                INVALID_AGENT_ROUTE,
                f"task_type={step.task_type} 不允许 agent={step.agent}",
            )
        if step.step_id in step.depends_on:
            raise MasterPlanValidationError(SELF_DEPENDENCY, "step 不得依赖自身")
        missing = [dependency for dependency in step.depends_on if dependency not in known]
        if missing:
            raise MasterPlanValidationError(
                INVALID_DEPENDENCY, f"依赖不存在: {', '.join(missing)}"
            )

    graph = {step.step_id: list(step.depends_on) for step in plan.steps}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visiting:
            raise MasterPlanValidationError(PLAN_CYCLE, "plan 存在循环依赖")
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency in graph[step_id]:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in graph:
        visit(step_id)
    return plan


__all__ = [
    "AGENT_ID_ALIASES",
    "ALLOWED_TASK_TYPES",
    "AVAILABLE_AGENTS",
    "TASK_ROUTE_MAP",
    "MasterPlanValidationError",
    "is_composite_customer_reply",
    "validate_master_plan",
]
