from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.core.skill.skill_registry import register_skill
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.modules.utils.llm_explain import llm_explain
from ecom_agent_matrix.platform.observability.context import TraceContext, trace_context
from ecom_agent_matrix.platform.observability.metrics import (
    REGISTRY,
    estimate_llm_cost,
    metrics,
    observed_workflow,
)
from ecom_agent_matrix.core.llm.providers.openai import OpenAIProvider
from ecom_agent_matrix.config.settings import settings


@register_skill
class _MetricsTimeoutSkill(BaseSkill):
    skill_name = "test_metrics_timeout"
    skill_desc = "metrics timeout"
    read_only = True
    side_effect = False
    risk_level = "low"
    timeout_seconds = 0.001

    async def run(self, params):
        await asyncio.sleep(1)
        return SkillResult(success=True)


@register_skill
class _MetricsSuccessSkill(BaseSkill):
    skill_name = "test_metrics_success"
    skill_desc = "metrics success"
    read_only = True
    side_effect = False
    risk_level = "low"

    async def run(self, params):
        return SkillResult(success=True, data={"ok": True})


def _value(name, labels):
    return REGISTRY.get_sample_value(name, labels) or 0


def test_http_and_agent_metrics_increment_without_high_cardinality_labels():
    before_http = _value("http_requests_total", {"method": "POST", "route": "/api/v1/tasks", "status_class": "2xx"})
    before_agent = _value("agent_tasks_total", {"agent": "data_query", "status": "success"})
    before_failure = _value("agent_tasks_total", {"agent": "data_query", "status": "failure"})
    metrics.observe_http("POST", "/api/v1/tasks", 200, 0.12)
    metrics.observe_agent("data_query", True, 0.05)
    metrics.observe_agent("data_query", False, 0.02)
    assert _value("http_requests_total", {"method": "POST", "route": "/api/v1/tasks", "status_class": "2xx"}) == before_http + 1
    assert _value("agent_tasks_total", {"agent": "data_query", "status": "success"}) == before_agent + 1
    assert _value("agent_tasks_total", {"agent": "data_query", "status": "failure"}) == before_failure + 1
    assert _value("http_request_duration_seconds_count", {"method": "POST", "route": "/api/v1/tasks"}) >= 1


def test_skill_executor_emits_timeout_and_normal_skill_metrics():
    before_timeout = _value("skill_executions_total", {"skill": "test_metrics_timeout", "status": "failure", "error_code": "TIMEOUT"})
    result = asyncio.run(exec_skill("test_metrics_timeout", {}))
    before_success = _value(
        "skill_executions_total",
        {"skill": "test_metrics_success", "status": "success", "error_code": "none"},
    )
    success = asyncio.run(exec_skill("test_metrics_success", {}))
    assert result.error_code == "TIMEOUT"
    assert success.success
    assert _value("skill_executions_total", {"skill": "test_metrics_timeout", "status": "failure", "error_code": "TIMEOUT"}) == before_timeout + 1
    assert _value(
        "skill_executions_total",
        {"skill": "test_metrics_success", "status": "success", "error_code": "none"},
    ) == before_success + 1


def test_approval_required_metric_label_is_bounded():
    before = _value("skill_executions_total", {"skill": "record_order_risk", "status": "failure", "error_code": "APPROVAL_REQUIRED"})
    metrics.observe_skill("record_order_risk", False, "APPROVAL_REQUIRED", 0.01)
    assert _value("skill_executions_total", {"skill": "record_order_risk", "status": "failure", "error_code": "APPROVAL_REQUIRED"}) == before + 1


def test_llm_metrics_tokens_cost_and_rule_fallback_behavior():
    price = {"openai:gpt-fixed": {"input_per_1m": 2.0, "output_per_1m": 4.0}}
    assert estimate_llm_cost("openai", "gpt-fixed", 1000, 500, price) == 0.004
    assert estimate_llm_cost("openai", "unknown", 1000, 500, price) is None
    before = _value("llm_calls_total", {"provider": "openai", "purpose": "planner", "status": "success"})
    metrics.observe_llm("openai", "planner", True, 0.2, 7, 3, 0.001)
    assert _value("llm_calls_total", {"provider": "openai", "purpose": "planner", "status": "success"}) == before + 1
    assert _value("llm_tokens_total", {"provider": "openai", "token_type": "prompt"}) >= 7

    async def fallback():
        with patch("ecom_agent_matrix.modules.utils.llm_explain.is_llm_configured", return_value=False):
            return await llm_explain(system_prompt="s", user_prompt="q", fallback="rules")

    calls_before = sum(sample.value for metric in REGISTRY.collect() if metric.name == "llm_calls" for sample in metric.samples if sample.name == "llm_calls_total")
    assert asyncio.run(fallback())[0] == "rules"
    calls_after = sum(sample.value for metric in REGISTRY.collect() if metric.name == "llm_calls" for sample in metric.samples if sample.name == "llm_calls_total")
    assert calls_after == calls_before


def test_real_provider_invocation_emits_usage_and_estimated_cost_metrics():
    class Response:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return False
        async def json(self, **_kwargs):
            return {
                "model": "gpt-fixed", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }

    class Session:
        def post(self, *_args, **_kwargs): return Response()

    before = _value("llm_calls_total", {"provider": "openai", "purpose": "planner", "status": "success"})

    async def scenario():
        with patch.object(settings, "OPENAI_API_KEY", "configured"), patch.object(
            settings, "OPENAI_CHAT_MODEL", "gpt-fixed"
        ), patch.object(
            settings, "LLM_PRICE_TABLE", {"openai:gpt-fixed": {"input_per_1m": 1, "output_per_1m": 2}}
        ), patch(
            "ecom_agent_matrix.core.llm.provider.get_http_session", new=AsyncMock(return_value=Session())
        ):
            with trace_context(TraceContext(workflow="planner")):
                return await OpenAIProvider()._chat_once(
                    user_prompt="private prompt", system_prompt="system", temperature=0.1,
                    max_tokens=20, mode="chat",
                )

    result = asyncio.run(scenario())
    assert result.total_tokens == 15
    assert _value("llm_calls_total", {"provider": "openai", "purpose": "planner", "status": "success"}) == before + 1
    assert _value("llm_estimated_cost_usd_total", {"provider": "openai"}) > 0


def test_metrics_render_prometheus_without_business_identifiers():
    rendered = metrics.render().decode()
    assert "http_requests_total" in rendered and "skill_executions_total" in rendered
    for secret in ("raw.jwt", "tenant-secret", "user-secret", "ORD-", "SKU-"):
        assert secret not in rendered


def test_metrics_endpoint_returns_prometheus_text_in_development():
    from ecom_agent_matrix.api.main import prometheus_metrics

    with patch.object(settings, "METRICS_AUTH_REQUIRED", False):
        response = asyncio.run(prometheus_metrics(None, None))
    assert response.status_code == 200
    assert b"http_requests_total" in response.body
    assert response.media_type.startswith("text/plain")


def test_workflow_exception_is_counted_as_failure():
    @observed_workflow("test_exception_workflow")
    async def explode():
        raise RuntimeError("business payload must not become a metric label")

    labels = {"workflow": "test_exception_workflow", "status": "failure"}
    before = _value("workflow_runs_total", labels)
    import pytest
    with pytest.raises(RuntimeError):
        asyncio.run(explode())
    assert _value("workflow_runs_total", labels) == before + 1
