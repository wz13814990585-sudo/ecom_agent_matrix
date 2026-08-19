"""Low-cardinality observability context; never used for business authorization."""
from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, replace
from typing import Iterator


def identity_hash(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:16] if value else ""


@dataclass(frozen=True)
class TraceContext:
    task_id: str = ""
    correlation_id: str = ""
    agent_id: str = ""
    workflow: str = ""
    skill_name: str = ""
    tenant_hash: str = ""
    user_hash: str = ""
    request_started_at: float = 0.0

    @classmethod
    def from_identity(cls, *, tenant_id: str = "", user_id: str = "", **fields):
        return cls(
            tenant_hash=identity_hash(tenant_id),
            user_hash=identity_hash(user_id),
            request_started_at=fields.pop("request_started_at", 0.0) or time.time(),
            **fields,
        )


@dataclass
class RequestPerformanceSummary:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    has_estimated_cost: bool = False

    def compact(self) -> dict:
        value = asdict(self)
        value.pop("has_estimated_cost")
        if not self.has_estimated_cost:
            value["estimated_cost_usd"] = None
        else:
            value["estimated_cost_usd"] = round(self.estimated_cost_usd, 8)
        return value


_trace: ContextVar[TraceContext] = ContextVar("trace_context", default=TraceContext())
_performance: ContextVar[RequestPerformanceSummary | None] = ContextVar(
    "request_performance", default=None
)
# Agent queues are consumed by long-lived asyncio tasks, so ContextVar values do
# not cross that queue boundary. This request-scoped map joins observability only
# by the existing root task_id; HTTP middleware owns its bounded lifecycle.
_request_performance: dict[str, RequestPerformanceSummary] = {}


def get_trace_context() -> TraceContext:
    return _trace.get()


def set_trace_context(context: TraceContext) -> Token:
    return _trace.set(context)


def update_trace_context(**updates) -> TraceContext:
    current = get_trace_context()
    allowed = {key: value for key, value in updates.items() if hasattr(current, key)}
    updated = replace(current, **allowed)
    _trace.set(updated)
    return updated


@contextmanager
def trace_context(context: TraceContext | None = None, **updates) -> Iterator[TraceContext]:
    base = context or get_trace_context()
    value = replace(base, **{key: value for key, value in updates.items() if hasattr(base, key)})
    token = _trace.set(value)
    perf_token = None
    if _performance.get() is None:
        perf_token = _performance.set(RequestPerformanceSummary())
    try:
        yield value
    finally:
        _trace.reset(token)
        if perf_token is not None:
            _performance.reset(perf_token)


def record_llm_usage(prompt: int, completion: int, total: int, cost: float | None) -> None:
    task_id = get_trace_context().task_id
    summary = _request_performance.get(task_id) if task_id else None
    if summary is None:
        summary = _performance.get()
    if summary is None:
        summary = RequestPerformanceSummary()
        _performance.set(summary)
    summary.llm_calls += 1
    summary.prompt_tokens += max(0, int(prompt or 0))
    summary.completion_tokens += max(0, int(completion or 0))
    summary.total_tokens += max(0, int(total or 0))
    if cost is not None:
        summary.estimated_cost_usd += max(0.0, float(cost))
        summary.has_estimated_cost = True


def get_performance_summary() -> dict:
    task_id = get_trace_context().task_id
    summary = _request_performance.get(task_id) if task_id else None
    if summary is None:
        summary = _performance.get()
    return summary.compact() if summary else RequestPerformanceSummary().compact()


def begin_request_performance(task_id: str) -> None:
    if task_id:
        _request_performance[task_id] = RequestPerformanceSummary()


def finish_request_performance(task_id: str) -> None:
    if task_id:
        _request_performance.pop(task_id, None)


__all__ = [
    "RequestPerformanceSummary", "TraceContext", "begin_request_performance",
    "finish_request_performance", "get_performance_summary",
    "get_trace_context", "identity_hash", "record_llm_usage", "set_trace_context",
    "trace_context", "update_trace_context",
]
