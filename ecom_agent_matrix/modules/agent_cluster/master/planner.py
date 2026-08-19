"""Typed Master Planner：规则 composite 优先，LLM 输出必须验证。"""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ecom_agent_matrix.config.constants import AGENT_EXEC, AGENT_QUERY, AGENT_RAG
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm import is_llm_configured, llm_chat, resolve_mode
from ecom_agent_matrix.modules.agent_cluster.master.policy import (
    MasterPlanValidationError,
    is_composite_customer_reply,
    validate_master_plan,
)
from ecom_agent_matrix.modules.agent_cluster.master.prompts import PLANNER_SYSTEM_PROMPT
from ecom_agent_matrix.modules.agent_cluster.master.schemas import MasterPlan, PlanStep
from ecom_agent_matrix.modules.agent_cluster.master.telemetry import MasterLLMTelemetry

def _query(task_input: dict) -> str:
    return str(
        task_input.get("query")
        or task_input.get("user_query")
        or task_input.get("text")
        or ""
    ).strip()


def build_composite_plan(task_input: dict) -> MasterPlan | None:
    """订单事实 + 政策知识 → CRM 回复的高置信 DAG 模板。"""
    query = _query(task_input)
    if not is_composite_customer_reply(query):
        return None
    plan = MasterPlan(
        decision="execute",
        confidence=0.98,
        reason_code="COMPOSITE_CUSTOMER_REPLY",
        planner_source="rules_composite",
        steps=[
            PlanStep(
                step_id="order_context",
                agent=AGENT_QUERY,
                task_type="order_query",
                payload={"query": query},
            ),
            PlanStep(
                step_id="policy_context",
                agent=AGENT_RAG,
                task_type="knowledge_qa",
                payload={"query": query},
            ),
            PlanStep(
                step_id="customer_reply",
                agent=AGENT_EXEC,
                task_type="customer_service",
                depends_on=["order_context", "policy_context"],
                payload={"query": query},
            ),
        ],
    )
    return validate_master_plan(plan)


def _clarify(reason_code: str, question: str) -> MasterPlan:
    return MasterPlan(
        decision="clarify",
        confidence=0.2,
        reason_code=reason_code,
        clarification_question=question,
        planner_source="deterministic_fallback",
        steps=[],
    )


def _extract_json(text: str) -> dict:
    raw = str(text or "").strip()
    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if block:
        raw = block.group(1)
    else:
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            raw = brace.group(0)
    return json.loads(raw)


class TypedMasterPlanner:
    async def plan(
        self,
        task_input: dict,
        telemetry: MasterLLMTelemetry,
    ) -> MasterPlan:
        composite = build_composite_plan(task_input)
        if composite is not None:
            return composite
        if not is_llm_configured():
            return _clarify(
                "UNKNOWN_NO_LLM",
                "请补充需要组合查询、知识检索或业务执行的具体目标。",
            )
        if not telemetry.start_call("planner"):
            return _clarify(
                "LLM_BUDGET_EXCEEDED",
                "当前复杂规划预算已用尽，请缩小请求范围后重试。",
            )

        recovery_context = task_input.get("_recovery_context")
        recovery_block = ""
        if isinstance(recovery_context, dict):
            recovery_block = (
                "\nCompact recovery context:\n"
                f"{json.dumps(recovery_context, ensure_ascii=False, default=str)[:1800]}\n"
            )
        prompt = (
            f"User request:\n{_query(task_input)[:1200]}\n"
            f"{recovery_block}\nReturn a validated dependency plan."
        )
        try:
            raw = await llm_chat(
                user_prompt=prompt,
                system_prompt=PLANNER_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=int(settings.MASTER_PLAN_MAX_TOKENS),
                mode=resolve_mode(settings.MASTER_PLAN_MODE),
            )
            telemetry.add_result("planner", raw)
            parsed = _extract_json(raw.content)
            parsed["planner_source"] = "llm"
            plan = MasterPlan.model_validate(parsed)
            return validate_master_plan(plan)
        except (json.JSONDecodeError, ValidationError, MasterPlanValidationError, TypeError, ValueError):
            return _clarify(
                "INVALID_LLM_PLAN",
                "复杂请求的执行计划未通过安全校验，请补充或简化需求。",
            )
        except Exception:
            return _clarify(
                "PLANNER_PROVIDER_ERROR",
                "复杂规划服务暂不可用，请稍后重试或简化需求。",
            )


typed_master_planner = TypedMasterPlanner()
