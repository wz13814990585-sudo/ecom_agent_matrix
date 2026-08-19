"""Master deterministic 路由：只判断，不执行任何 I/O。"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm import is_llm_configured
from ecom_agent_matrix.modules.agent_cluster.master_planner import TASK_ROUTE_MAP


class MasterRouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    mode: Literal["fast_path", "planner", "clarify"]
    task_type: str | None = None
    target_agents: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    reason_code: str
    source: str = "rules"


_RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "knowledge_qa",
        "RULE_KNOWLEDGE",
        re.compile(
            r"退款规则|退货政策|店铺规则|运营手册|FAQ|知识库|怎么退|"
            r"refund\s+policy|return\s+policy|store\s+policy",
            re.I,
        ),
    ),
    (
        "order_query",
        "RULE_ORDER_QUERY",
        re.compile(
            r"(?:ORD[-_][A-Z0-9_-]+|\d{10,20}).*(?:状态|物流|发货|订单)|"
            r"(?:状态|物流|发货|查询|查一下).*(?:ORD[-_][A-Z0-9_-]+|\d{10,20})|"
            r"查询订单|订单查询|订单状态|订单数据|order\s+status|tracking",
            re.I,
        ),
    ),
    (
        "stock_analysis",
        "RULE_STOCK",
        re.compile(r"库存|备货|补货|缺货|stock|inventory|replenish", re.I),
    ),
    (
        "ad_optimize",
        "RULE_AD_OPTIMIZE",
        re.compile(r"优化.*广告|广告.*优化|调整出价|调出价|投放优化|optimi[sz]e.*(?:ad|campaign)", re.I),
    ),
    (
        "ops_report",
        "RULE_REPORT",
        re.compile(r"生成.*(?:运营)?(?:日报|周报|报表|报告)|运营日报|运营周报|ops\s+report", re.I),
    ),
    (
        "customer_service",
        "RULE_CRM",
        re.compile(r"回复.*(?:客户|顾客|买家)|(?:客服|售后).*(?:回复|处理)|帮我回复|customer\s+(?:reply|support)", re.I),
    ),
    (
        "risk_control",
        "RULE_RISK",
        re.compile(r"触发.*风控|风控扫描|风险拦截|risk\s+check", re.I),
    ),
    (
        "competitor_watch",
        "RULE_COMPETITOR",
        re.compile(r"竞品|比价|价格对比|竞价对比|跟价|competitor|price\s+compar", re.I),
    ),
    (
        "data_check",
        "RULE_DATA_CHECK",
        re.compile(r"数据校验|完整性检查|脏数据|孤儿订单|执行\s*sql|跑\s*sql|data\s+check", re.I),
    ),
    (
        "ad_query",
        "RULE_AD_QUERY",
        re.compile(r"查询.*广告|广告数据|投放数据|广告消耗|ad\s+(?:data|spend)", re.I),
    ),
    (
        "social_marketing",
        "RULE_SOCIAL",
        re.compile(r"生成.*(?:社媒|营销|发帖).*文案|社媒文案|social\s+(?:copy|caption)", re.I),
    ),
    (
        "goods_catalog",
        "RULE_GOODS_CATALOG",
        re.compile(r"商品总数|商品数量|列出.*商品|全部商品|商品目录|product\s+catalog|list\s+all\s+products", re.I),
    ),
    (
        "goods_search",
        "RULE_GOODS_SEARCH",
        re.compile(r"搜索.*商品|查找.*商品|找.*(?:商品|款式)|商品搜索|search.*product", re.I),
    ),
)


def _query(task_input: dict) -> str:
    for key in ("query", "user_query", "text", "message", "content"):
        value = task_input.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def route_master_task(task_input: dict) -> MasterRouteDecision:
    """返回单意图 Fast Path、复杂 Planner 或无模型 Clarify 决策。"""
    explicit = str(task_input.get("task_type") or "").strip()
    if explicit in TASK_ROUTE_MAP:
        enabled = settings.MASTER_FAST_PATH_ENABLED
        return MasterRouteDecision(
            mode="fast_path" if enabled else "planner",
            task_type=explicit,
            target_agents=list(TASK_ROUTE_MAP[explicit]) if enabled else [],
            confidence=1.0,
            reason_code="EXPLICIT_TASK_TYPE" if enabled else "FAST_PATH_DISABLED",
            source="explicit" if enabled else "settings",
        )

    query = _query(task_input)
    matches = [rule for rule in _RULES if rule[2].search(query)]
    matched_types = {rule[0] for rule in matches}
    if len(matched_types) > 1:
        return MasterRouteDecision(
            mode="planner" if is_llm_configured() else "clarify",
            confidence=0.4,
            reason_code="AMBIGUOUS",
            source="rules",
        )

    if matches:
        task_type, reason_code, _pattern = matches[0]
        enabled = settings.MASTER_FAST_PATH_ENABLED
        return MasterRouteDecision(
            mode="fast_path" if enabled else "planner",
            task_type=task_type,
            target_agents=list(TASK_ROUTE_MAP[task_type]) if enabled else [],
            confidence=0.95,
            reason_code=reason_code if enabled else "FAST_PATH_DISABLED",
            source="rules" if enabled else "settings",
        )

    if is_llm_configured():
        return MasterRouteDecision(
            mode="planner",
            confidence=0.2,
            reason_code="UNKNOWN",
            source="config",
        )
    return MasterRouteDecision(
        mode="clarify",
        confidence=0.2,
        reason_code="UNKNOWN",
        source="rules",
    )


__all__ = ["MasterRouteDecision", "route_master_task"]
