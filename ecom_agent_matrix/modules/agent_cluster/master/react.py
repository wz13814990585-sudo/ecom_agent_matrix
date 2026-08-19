"""Recovery-only ReAct；正常 DAG execution 不调用。"""
from __future__ import annotations

import json

from pydantic import ValidationError

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm import is_llm_configured, llm_chat, resolve_mode
from ecom_agent_matrix.modules.agent_cluster.master.prompts import RECOVERY_SYSTEM_PROMPT
from ecom_agent_matrix.modules.agent_cluster.master.schemas import (
    PlanExecutionResult,
    RecoveryDecision,
)
from ecom_agent_matrix.modules.agent_cluster.master.telemetry import MasterLLMTelemetry
from ecom_agent_matrix.platform.observability.context import trace_context

_RECOVERABLE = frozenset({"TIMEOUT", "AGENT_FAILED", "STEP_EXECUTION_ERROR"})


class RecoveryController:
    async def run(
        self,
        execution: PlanExecutionResult,
        telemetry: MasterLLMTelemetry,
    ) -> RecoveryDecision | None:
        if execution.all_success:
            return None
        failed = [
            result
            for result in execution.step_results.values()
            if result.status == "FAILED" and result.error_code in _RECOVERABLE
        ]
        if not failed:
            return None
        if not is_llm_configured():
            return RecoveryDecision(
                action="finish", reason_code="RECOVERY_LLM_UNAVAILABLE"
            )

        compact = [
            {
                "step_id": result.step_id,
                "agent": result.agent,
                "task_type": result.task_type,
                "error_code": result.error_code,
            }
            for result in failed
        ]
        for _ in range(max(0, int(settings.MASTER_RECOVERY_MAX_STEPS))):
            if not telemetry.start_call("recovery"):
                return RecoveryDecision(
                    action="finish", reason_code="LLM_BUDGET_EXCEEDED"
                )
            try:
                with trace_context(workflow="recovery"):
                    raw = await llm_chat(
                        user_prompt=f"Failed plan steps:\n{json.dumps(compact, ensure_ascii=False)}",
                        system_prompt=RECOVERY_SYSTEM_PROMPT,
                        temperature=0.1,
                        max_tokens=int(settings.MASTER_REACT_MAX_TOKENS),
                        mode=resolve_mode(settings.MASTER_REACT_MODE),
                    )
                telemetry.add_result("recovery", raw)
                parsed = json.loads(raw.content)
                decision = RecoveryDecision.model_validate(parsed)
                if decision.action == "retry_agent":
                    target = execution.step_results.get(decision.step_id)
                    if target is None or target.agent == "biz_exec":
                        return RecoveryDecision(
                            action="finish",
                            reason_code="UNSAFE_RECOVERY_RETRY_REJECTED",
                        )
                return decision
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
                continue
            except Exception:
                return RecoveryDecision(
                    action="finish", reason_code="RECOVERY_PROVIDER_ERROR"
                )
        return RecoveryDecision(action="finish", reason_code="RECOVERY_EXHAUSTED")


recovery_controller = RecoveryController()
