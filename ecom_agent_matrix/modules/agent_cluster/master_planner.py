"""Master Agent 任务规划：只做规划 / 分发 / 聚合，不写业务逻辑。

子 Agent 按任务类型只有 3 个：
- data_query：只读数据查询（广告/订单/库存/竞品/商品目录）
- biz_exec：写操作与产物生成（调出价、风控、报表、社媒）
- knowledge_rag：店铺规则 / 运营手册，不碰业务表
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

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
from ecom_agent_matrix.core.llm import is_llm_configured, llm_chat, resolve_mode
from ecom_agent_matrix.core.logging_config import setup_logger

logger = setup_logger("agent.master_planner")

AVAILABLE_AGENTS: dict[str, str] = {
    AGENT_QUERY: (
        "只读数据查询：广告数据、订单、库存、竞品价、商品目录/SKU。"
        "内部自己解析商品名→SKU 并调用 DB Skill，不改业务状态。"
    ),
    AGENT_EXEC: (
        "写操作与产物：调整广告出价、触发风控、生成运营报表、社媒文案。"
        "会变更状态或产出运营物料。"
    ),
    AGENT_RAG: "文档知识库：店铺规则、运营手册、FAQ。只检索向量知识，不查业务表。",
}

# 细粒度 task_type → 唯一子 Agent（Master 不再按实体拆步）
TASK_ROUTE_MAP: dict[str, list[str]] = {
    "customer_service": [AGENT_EXEC],  # 客服回复 / 售后处理 → CRM workflow
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

TASK_KEYWORDS: dict[str, list[str]] = {
    "knowledge_qa": [
        "知识", "问答", "怎么用", "介绍一下", "材质", "规格说明",
        "店铺规则", "运营手册", "退货政策", "退款规则", "怎么退",
        "knowledge", "what is", "how to use", "faq", "policy", "手册", "规则",
    ],
    "goods_catalog": [
        "有多少商品", "多少商品", "商品数量", "商品总数", "列出商品",
        "有哪些商品", "全部商品", "所有商品", "商品列表", "商品目录",
        "数据库里", "库里有", "查询数据库", "查库",
        "how many products", "list products", "product catalog",
    ],
    "order_query": [
        "查订单", "查询订单", "订单查询", "订单数据", "订单状态", "物流", "发货", "查一下订单",
        "order status", "tracking", "shipment",
    ],
    "ad_query": [
        "广告数据", "投放数据", "广告消耗", "查广告", "广告报表查询",
        "ad data", "ad spend", "campaign stats",
    ],
    "ad_optimize": [
        "调整出价", "调出价", "投放优化", "优化广告", "广告竞价",
        "roi", "campaign", "ad", "optimize", "ppc", "ads",
    ],
    "goods_search": [
        "商品", "搜索", "找", "推荐", "款式", "连衣裙", "背包",
        "product", "search", "find", "recommend", "catalog",
    ],
    "stock_analysis": [
        "库存", "备货", "补货", "缺货", "forecast",
        "stock", "inventory", "replenish",
    ],
    "social_marketing": [
        "社媒", "文案", "营销", "发帖", "instagram", "tiktok",
        "social", "marketing", "copy", "caption",
    ],
    "competitor_watch": [
        "竞品", "对手", "监控", "预警", "比价", "竞价对比", "价格对比",
        "竞价", "跟价", "temu", "amazon",
        "competitor", "monitor", "watch", "price compare", "price comparison",
    ],
    "data_check": [
        "数据校验", "完整性", "脏数据", "主数据", "孤儿订单",
        "跑sql", "执行sql", "data check", "integrity", "orphan", "校验",
    ],
    "ops_report": [
        "报表", "日报", "周报", "运营报告", "gmv", "汇总",
        "report", "dashboard", "简报", "生成报表",
    ],
    "risk_control": [
        "触发风控", "风控扫描", "风险拦截", "risk check", "风控",
    ],
    "customer_service": [
        "退款", "退货", "客服", "投诉", "售后",
        "refund", "return", "customer", "support",
    ],
}

_AGENT_LIST_PROMPT = "\n".join(f"- {aid}: {desc}" for aid, desc in AVAILABLE_AGENTS.items())

PLANNER_SYSTEM_PROMPT = f"""You are the Master Planning Agent for a cross-border ecommerce multi-agent system.
You ONLY plan, dispatch and aggregate. Do NOT implement business logic.

Available agents (use EXACT ids):
{_AGENT_LIST_PROMPT}

Return ONLY valid JSON, no markdown:
{{
  "decision": "dispatch or clarify",
  "agents": ["agent_id1"],
  "confidence": 0.85,
  "reasoning": "brief decision reason in one sentence",
  "clarification_question": "question when decision is clarify"
}}

Rules:
- pick 1 agent in most cases; at most 2 (query then exec) if user both reads AND mutates
- 查数据 / 有多少 / 库存 / 订单 / 竞品价 / 广告数据 → data_query
- 调出价 / 改库存 / 触发风控 / 生成报表 / 写文案 → biz_exec
- 店铺规则 / 运营手册 / FAQ / 退款规则 → knowledge_rag
- 客服回复 / 售后处理 → biz_exec（CRM workflow）
- 无法可靠判断 → decision=clarify，不调用任何 Agent
- never pick entity agents like goods_lookup / stock_agent / ad_optimizer
- use exact agent ids from the list above
"""

REACT_SYSTEM_PROMPT = f"""You are the Master ReAct controller.
Decide the NEXT single action. You do planning/dispatch only.

Available agents (use EXACT ids):
{_AGENT_LIST_PROMPT}

Return ONLY valid JSON:
{{
  "thought": "brief reasoning",
  "action": "call_agent" or "finish",
  "agent": "agent_id if call_agent else empty",
  "skill": "",
  "payload": {{}},
  "final_answer": "summary when finish"
}}

Rules:
- call only ONE agent per step
- after data_query / biz_exec / knowledge_rag returns → finish (they resolve SKU internally)
- do NOT call skills for DB or ads; those belong to sub-agents
- payload may include query, sku, task_type
"""

# 旧实体 Agent / skill 名 → 任务类型 Agent
_AGENT_ID_ALIASES: dict[str, str] = {
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


@dataclass
class PlanResult:
    sub_tasks: list[dict]
    plan_confidence: float
    reasoning: str
    planner: str = "llm"
    raw_llm: str = ""
    decision: str = "dispatch"  # dispatch | clarify
    clarification_question: str = ""


@dataclass
class ReactDecision:
    thought: str
    action: str  # call_agent | call_skill | finish
    agent: str = ""
    skill: str = ""
    payload: dict = field(default_factory=dict)
    final_answer: str = ""
    confidence: float = 0.7
    source: str = "rules"
    reasoning_content: str = ""


def _build_memory_context(memory_hits: list[dict]) -> list[dict]:
    return [
        {
            "content": hit.get("content"),
            "meta": hit.get("meta"),
            "distance": hit.get("distance"),
            "confidence": (hit.get("meta") or {}).get("confidence"),
        }
        for hit in memory_hits
    ]


def _truncate_reasoning(text: str, limit: int | None = None) -> str:
    limit = int(limit if limit is not None else settings.MASTER_REASONING_STORE_CHARS)
    raw = (text or "").strip()
    if limit <= 0 or len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 1)] + "…"


def _extract_query(task_input: dict) -> str:
    for key in ("query", "user_query", "text", "message", "content", "product_name"):
        val = task_input.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _enrich_payload(task_input: dict, memory_hits: list[dict], planner: str, extra: dict | None = None) -> dict:
    payload = {
        **task_input,
        "_memory_context": _build_memory_context(memory_hits),
        "_planner": planner,
    }
    if extra:
        payload.update(extra)
    return payload


def _extract_json(text: str) -> dict:
    text = text.strip()
    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if block:
        text = block.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    return json.loads(text)


def _validate_agents(agent_ids: list[str]) -> list[str]:
    valid = []
    for aid in agent_ids:
        key = str(aid).strip()
        key = _AGENT_ID_ALIASES.get(key, key)
        if key in AVAILABLE_AGENTS and key not in valid:
            valid.append(key)
    return valid


def infer_task_type_from_query(query: str) -> str | None:
    """关键词分类器：从 query 推断细粒度 task_type。"""
    text = (query or "").strip()
    if not text:
        return None

    from ecom_agent_matrix.modules.parsers.goods import is_catalog_query

    if is_catalog_query(text):
        return "goods_catalog"

    lower = text.lower()
    if re.search(r"竞价对比|价格对比|比价|竞品价|跟价", text) or re.search(
        r"price\s*compar", lower
    ):
        return "competitor_watch"
    if "竞价" in text and not re.search(r"广告|投放|ppc|campaign", lower):
        return "competitor_watch"
    if re.search(r"调整出价|调出价|优化广告", text):
        return "ad_optimize"
    if re.search(r"广告数据|投放数据|查广告", text) and not re.search(r"优化|调整", text):
        return "ad_query"
    if re.search(r"触发风控|风控扫描", text):
        return "risk_control"
    if re.search(r"店铺规则|运营手册|退货政策|退款规则|怎么退", text):
        return "knowledge_qa"

    scores: dict[str, int] = {}
    for task_type, keywords in TASK_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in lower)
        if score:
            scores[task_type] = score

    if not scores:
        return None
    return max(scores, key=scores.get)


def plan_sub_tasks_clarify(task_input: dict, memory_hits: list[dict], reason: str) -> PlanResult:
    """无法可靠识别意图时请求澄清，不向子 Agent 分发。"""
    return PlanResult(
        sub_tasks=[],
        plan_confidence=0.3,
        reasoning=reason,
        planner="clarify",
        decision="clarify",
        clarification_question="请说明您要查询的数据、咨询的店铺规则，或需要执行的业务操作。",
    )


def plan_sub_tasks_rag_default(task_input: dict, memory_hits: list[dict], reason: str) -> PlanResult:
    """兼容旧调用名；P0 起未知意图改为 clarify。"""
    return plan_sub_tasks_clarify(task_input, memory_hits, reason)


# 兼容旧测试名
def plan_sub_tasks_crm_default(task_input: dict, memory_hits: list[dict], reason: str) -> PlanResult:
    return plan_sub_tasks_clarify(task_input, memory_hits, reason)


def plan_sub_tasks_rules(
    task_input: dict,
    memory_hits: list[dict],
    *,
    task_type: str | None = None,
    planner: str = "rules",
) -> PlanResult:
    resolved = task_type or task_input.get("task_type")
    if resolved and resolved in TASK_ROUTE_MAP:
        agents = list(TASK_ROUTE_MAP[resolved])
        extra = {"mode": "catalog"} if resolved == "goods_catalog" else None
        payload = _enrich_payload(task_input, memory_hits, planner, extra=extra)
        sub_tasks = [{"target_agent": agent_id, "payload": payload} for agent_id in agents]
        return PlanResult(
            sub_tasks=sub_tasks,
            plan_confidence=0.7,
            reasoning=f"规则路由 task_type={resolved} → {agents}",
            planner=planner,
        )
    return plan_sub_tasks_keyword(task_input, memory_hits)


def plan_sub_tasks_keyword(task_input: dict, memory_hits: list[dict]) -> PlanResult:
    query = _extract_query(task_input)
    inferred = infer_task_type_from_query(query) if query else None
    if inferred and inferred in TASK_ROUTE_MAP:
        agents = list(TASK_ROUTE_MAP[inferred])
        payload = _enrich_payload(
            task_input,
            memory_hits,
            "keyword",
            extra={
                "_inferred_task_type": inferred,
                **({"mode": "catalog"} if inferred == "goods_catalog" else {}),
            },
        )
        sub_tasks = [{"target_agent": agent_id, "payload": payload} for agent_id in agents]
        return PlanResult(
            sub_tasks=sub_tasks,
            plan_confidence=0.55,
            reasoning=f"关键词推断 task_type={inferred} → {agents}",
            planner="keyword",
        )
    return plan_sub_tasks_clarify(
        task_input,
        memory_hits,
        reason="关键词无法可靠识别意图，需要用户澄清",
    )


async def plan_sub_tasks_llm(task_input: dict, memory_hits: list[dict]) -> PlanResult:
    """
    初始意图规划：
    1. LLM 在 {data_query, biz_exec, knowledge_rag} 中选择
    2. 失败/低置信 → 关键词
    3. 仍无法识别 → clarify
    """
    query = _extract_query(task_input)

    explicit_task_type = str(task_input.get("task_type") or "").strip()
    if explicit_task_type in TASK_ROUTE_MAP:
        return plan_sub_tasks_rules(
            task_input,
            memory_hits,
            task_type=explicit_task_type,
            planner="rules_explicit_task_type",
        )

    # 关键业务路由使用确定性规则，避免知识问答、CRM、订单查询相互混淆。
    inferred = infer_task_type_from_query(query)
    if inferred is None:
        return plan_sub_tasks_clarify(
            task_input,
            memory_hits,
            reason="请求缺少可可靠识别的业务意图",
        )
    if inferred in {"knowledge_qa", "customer_service", "order_query"}:
        return plan_sub_tasks_rules(
            task_input,
            memory_hits,
            task_type=inferred,
            planner="rules_critical_intent",
        )

    if not is_llm_configured():
        result = plan_sub_tasks_keyword(task_input, memory_hits)
        result.reasoning = "未配置 API Key，使用关键词路由或请求澄清"
        if result.planner == "keyword":
            result.planner = "keyword_no_api_key"
        elif result.planner == "clarify":
            result.planner = "clarify_no_api_key"
        return result

    memory_snippet = json.dumps(_build_memory_context(memory_hits), ensure_ascii=False)[:1500]
    user_prompt = (
        f"User request (natural language):\n{query or json.dumps(task_input, ensure_ascii=False)}\n\n"
        f"Historical memory (may be empty):\n{memory_snippet}\n\n"
        "Choose sub-agents (ordered for ReAct) and confidence."
    )

    try:
        raw = await llm_chat(
            user_prompt=user_prompt,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=int(settings.MASTER_PLAN_MAX_TOKENS),
            mode=resolve_mode(settings.MASTER_PLAN_MODE),
        )
        parsed = _extract_json(raw.content)
        decision = str(parsed.get("decision") or "dispatch").strip().lower()
        if decision == "clarify":
            result = plan_sub_tasks_clarify(
                task_input,
                memory_hits,
                reason=_truncate_reasoning(str(parsed.get("reasoning") or "LLM 请求澄清")),
            )
            result.planner = "llm_clarify"
            result.plan_confidence = float(parsed.get("confidence", 0.0))
            result.clarification_question = str(
                parsed.get("clarification_question") or result.clarification_question
            )
            result.raw_llm = raw.content
            return result
        agents = _validate_agents(parsed.get("agents", []))
        confidence = float(parsed.get("confidence", 0.0))
        # 仅保留模型显式给出的简短 reason；内部 reasoning_content 不进入计划或记忆。
        reasoning = _truncate_reasoning(str(parsed.get("reasoning", "")))

        if not agents:
            raise ValueError("LLM 未返回有效 agent")

        if AGENT_RAG in agents and infer_task_type_from_query(query) != "knowledge_qa":
            result = plan_sub_tasks_clarify(
                task_input,
                memory_hits,
                reason="LLM 建议知识库，但请求缺少明确知识库意图",
            )
            result.planner = "clarify_rag_guard"
            result.raw_llm = raw.content
            return result

        if confidence < settings.MASTER_PLAN_MIN_CONFIDENCE:
            fallback = plan_sub_tasks_keyword(task_input, memory_hits)
            fallback.planner = "keyword_low_confidence"
            fallback.reasoning = (
                f"LLM 置信度过低({confidence:.2f}<{settings.MASTER_PLAN_MIN_CONFIDENCE})，"
                f"回退关键词路由。LLM reason: {reasoning}"
            )
            fallback.raw_llm = raw.content
            return fallback

        payload = _enrich_payload(
            task_input,
            memory_hits,
            "llm",
            extra={"_plan_reasoning": reasoning, "_plan_confidence": confidence},
        )
        sub_tasks = [{"target_agent": aid, "payload": payload} for aid in agents]
        return PlanResult(
            sub_tasks=sub_tasks,
            plan_confidence=confidence,
            reasoning=reasoning,
            planner="llm",
            raw_llm=raw.content,
        )
    except Exception as exc:
        fallback = plan_sub_tasks_keyword(task_input, memory_hits)
        fallback.planner = "keyword_llm_fallback"
        fallback.reasoning = f"LLM 规划失败({exc})，回退关键词路由或请求澄清"
        return fallback


def react_decide_rules(
    working: dict,
    observations: list[dict],
    suggested_agents: list[str],
) -> ReactDecision:
    """规则 ReAct：Master 只调度 Query / Exec / RAG，SKU 解析在 Query 内部完成。"""
    query = _extract_query(working)
    suggested = _validate_agents(suggested_agents)
    called = {_AGENT_ID_ALIASES.get(o.get("agent"), o.get("agent")) for o in observations}

    if observations:
        last = observations[-1]
        last_ok = last.get("success")
        last_data = last.get("data") or {}
        last_agent = _AGENT_ID_ALIASES.get(last.get("agent"), last.get("agent"))

        if not last_ok:
            return ReactDecision(
                thought="子 Agent 失败，结束",
                action="finish",
                final_answer=last.get("error_msg") or "子任务失败",
                source="rules",
            )

        # 建议序列还有下一步（例如先查数据再执行变更）
        for aid in suggested:
            if aid not in called:
                return ReactDecision(
                    thought=f"按任务类型调用下一 Agent: {aid}",
                    action="call_agent",
                    agent=aid,
                    payload={**working, "query": query},
                    source="rules",
                )

        summary = (
            last_data.get("summary")
            or last_data.get("advice")
            or last_data.get("answer")
            or "任务完成"
        )
        return ReactDecision(
            thought=f"{last_agent} 已完成，结束",
            action="finish",
            final_answer=str(summary),
            source="rules",
        )

    if not suggested:
        return ReactDecision(
            thought="缺少可靠的目标 Agent，需要用户澄清",
            action="finish",
            final_answer="请补充您想查询或处理的具体业务内容。",
            source="rules",
        )

    first = suggested[0]
    payload = {**working, "query": query}
    if first == AGENT_QUERY and infer_task_type_from_query(query) == "goods_catalog":
        payload["mode"] = "catalog"
    return ReactDecision(
        thought=f"按任务类型调用 {first}",
        action="call_agent",
        agent=first,
        payload=payload,
        source="rules",
    )


async def react_decide(
    working: dict,
    observations: list[dict],
    suggested_agents: list[str],
) -> ReactDecision:
    if not is_llm_configured():
        return react_decide_rules(working, observations, suggested_agents)

    user_prompt = (
        f"Working context:\n{json.dumps(working, ensure_ascii=False)[:2000]}\n\n"
        f"Suggested agents (ordered):\n{suggested_agents}\n\n"
        f"Observations so far:\n{json.dumps(observations, ensure_ascii=False)[:2500]}\n\n"
        "Decide the next single action."
    )
    try:
        raw = await llm_chat(
            user_prompt=user_prompt,
            system_prompt=REACT_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=int(settings.MASTER_REACT_MAX_TOKENS),
            mode=resolve_mode(settings.MASTER_REACT_MODE),
        )
        parsed = _extract_json(raw.content)
        action = str(parsed.get("action", "")).strip()
        agent = str(parsed.get("agent", "")).strip()
        reasoning_content = _truncate_reasoning(raw.reasoning_content or "")
        if action == "finish":
            return ReactDecision(
                thought=str(parsed.get("thought", "")),
                action="finish",
                final_answer=str(parsed.get("final_answer", "done")),
                confidence=0.85,
                source="llm",
                reasoning_content=reasoning_content,
            )
        payload = parsed.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        merged = {**working, **payload}
        if action == "call_agent":
            agent = _validate_agents([agent])[0] if _validate_agents([agent]) else ""
        if action == "call_agent" and agent in AVAILABLE_AGENTS:
            decision = ReactDecision(
                thought=str(parsed.get("thought", "")),
                action="call_agent",
                agent=agent,
                payload=merged,
                confidence=0.85,
                source="llm",
                reasoning_content=reasoning_content,
            )
            return _guard_react_decision(decision, working, observations, suggested_agents)
        raise ValueError(f"非法 ReAct 动作: action={action} agent={agent}")
    except Exception as exc:
        logger.warning(
            "react_decide_fallback",
            extra={
                "event": "react_decide_fallback",
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        decision = react_decide_rules(working, observations, suggested_agents)
        decision.thought = f"[llm_fallback:{type(exc).__name__}] {decision.thought}"
        return decision


def _guard_react_decision(
    decision: ReactDecision,
    working: dict,
    observations: list[dict],
    suggested_agents: list[str],
) -> ReactDecision:
    """纠正 LLM 误调旧实体 Agent id。"""
    if decision.action != "call_agent":
        return decision
    mapped = _validate_agents([decision.agent])
    if mapped:
        if mapped[0] != decision.agent:
            decision.thought = f"[guard] 实体 Agent 映射为任务类型 {mapped[0]}。原：{decision.thought}"
            decision.source = f"{decision.source}+guard"
        decision.agent = mapped[0]
        return decision
    fallback = react_decide_rules(working, observations, suggested_agents)
    fallback.thought = f"[guard] 非法 agent={decision.agent}，改用规则。原：{decision.thought}"
    fallback.source = f"{decision.source}+guard"
    return fallback


def merge_observation_into_working(working: dict, observation: dict) -> dict:
    updated = dict(working)
    data = observation.get("data") or {}
    if data.get("best_sku"):
        updated["sku"] = data["best_sku"]
        updated["best_sku"] = data["best_sku"]
        updated["target_sku"] = data["best_sku"]
    if data.get("sku"):
        updated["sku"] = data["sku"]
    if data.get("candidates"):
        updated["candidates"] = data["candidates"]
    kind = data.get("query_kind") or data.get("exec_kind")
    if kind:
        updated["_last_kind"] = kind
    return updated


def compute_memory_confidence(plan_confidence: float, final_result: dict[str, Any]) -> float:
    score = plan_confidence
    if final_result.get("timed_out"):
        score *= 0.2
    elif not final_result.get("all_success"):
        score *= 0.5
    else:
        expected = max(final_result.get("expected", 1), 1)
        received = final_result.get("received", 0)
        score *= received / expected
    return round(min(max(score, 0.0), 1.0), 3)


def should_save_to_memory(plan_confidence: float, final_result: dict[str, Any]) -> tuple[bool, float]:
    mem_conf = compute_memory_confidence(plan_confidence, final_result)
    ok = (
        final_result.get("all_success")
        and not final_result.get("timed_out")
        and mem_conf >= settings.MASTER_MEMORY_MIN_CONFIDENCE
    )
    return ok, mem_conf
