"""Central Prometheus facade with bounded labels."""
from __future__ import annotations

from functools import wraps
import time

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

REGISTRY = CollectorRegistry()


class Metrics:
    def __init__(self):
        r = REGISTRY
        self.http_requests = Counter("http_requests_total", "HTTP requests", ["method", "route", "status_class"], registry=r)
        self.http_duration = Histogram("http_request_duration_seconds", "HTTP latency", ["method", "route"], registry=r)
        self.agent_tasks = Counter("agent_tasks_total", "Agent tasks", ["agent", "status"], registry=r)
        self.agent_duration = Histogram("agent_task_duration_seconds", "Agent latency", ["agent"], registry=r)
        self.workflow_runs = Counter("workflow_runs_total", "Workflow runs", ["workflow", "status"], registry=r)
        self.workflow_duration = Histogram("workflow_duration_seconds", "Workflow latency", ["workflow"], registry=r)
        self.skill_executions = Counter("skill_executions_total", "Skill executions", ["skill", "status", "error_code"], registry=r)
        self.skill_duration = Histogram("skill_duration_seconds", "Skill latency", ["skill"], registry=r)
        self.llm_calls = Counter("llm_calls_total", "Real provider invocations", ["provider", "purpose", "status"], registry=r)
        self.llm_tokens = Counter("llm_tokens_total", "LLM tokens", ["provider", "token_type"], registry=r)
        self.llm_duration = Histogram("llm_request_duration_seconds", "LLM latency", ["provider", "purpose"], registry=r)
        self.llm_cost = Counter("llm_estimated_cost_usd_total", "Estimated LLM cost", ["provider"], registry=r)
        self.external_retries = Counter("external_retries_total", "External retries", ["component", "reason"], registry=r)
        self.rate_limit_rejections = Counter("rate_limit_rejections_total", "Rate-limit rejections", ["route"], registry=r)

    def observe_http(self, method: str, route: str, status: int, seconds: float) -> None:
        self.http_requests.labels(method.upper(), route, f"{int(status) // 100}xx").inc()
        self.http_duration.labels(method.upper(), route).observe(max(0.0, seconds))

    def observe_agent(self, agent: str, success: bool, seconds: float) -> None:
        self.agent_tasks.labels(agent, "success" if success else "failure").inc()
        self.agent_duration.labels(agent).observe(max(0.0, seconds))

    def observe_workflow_result(self, workflow: str, success: bool, seconds: float) -> None:
        self.workflow_runs.labels(workflow, "success" if success else "failure").inc()
        self.workflow_duration.labels(workflow).observe(max(0.0, seconds))

    def observe_skill(self, skill: str, success: bool, error_code: str, seconds: float) -> None:
        self.skill_executions.labels(skill, "success" if success else "failure", error_code or "none").inc()
        self.skill_duration.labels(skill).observe(max(0.0, seconds))

    def observe_llm(self, provider: str, purpose: str, success: bool, seconds: float, prompt: int = 0, completion: int = 0, cost: float | None = None) -> None:
        self.llm_calls.labels(provider, purpose, "success" if success else "failure").inc()
        self.llm_duration.labels(provider, purpose).observe(max(0.0, seconds))
        if prompt:
            self.llm_tokens.labels(provider, "prompt").inc(max(0, prompt))
        if completion:
            self.llm_tokens.labels(provider, "completion").inc(max(0, completion))
        if cost is not None:
            self.llm_cost.labels(provider).inc(max(0.0, cost))

    def render(self) -> bytes:
        return generate_latest(REGISTRY)


metrics = Metrics()


def estimate_llm_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int, price_table: dict | None) -> float | None:
    pricing = (price_table or {}).get(f"{provider}:{model}")
    if not isinstance(pricing, dict):
        return None
    try:
        input_price = float(pricing["input_per_1m"])
        output_price = float(pricing["output_per_1m"])
    except (KeyError, TypeError, ValueError):
        return None
    return (max(0, prompt_tokens) * input_price + max(0, completion_tokens) * output_price) / 1_000_000


def observed_workflow(name: str):
    def decorator(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            started = time.perf_counter()
            success = False
            from .context import trace_context
            try:
                with trace_context(workflow=name):
                    result = await func(*args, **kwargs)
                success = bool(getattr(result, "success", False))
                return result
            finally:
                metrics.observe_workflow_result(name, success, time.perf_counter() - started)
        return wrapped
    return decorator


__all__ = ["REGISTRY", "Metrics", "estimate_llm_cost", "metrics", "observed_workflow"]
