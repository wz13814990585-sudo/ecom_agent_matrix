from __future__ import annotations

import json
import logging

from ecom_agent_matrix.core.logging_config import JsonFormatter
from ecom_agent_matrix.platform.observability.context import (
    TraceContext, begin_request_performance, finish_request_performance,
    get_performance_summary, get_trace_context, identity_hash, record_llm_usage,
    trace_context,
)
from ecom_agent_matrix.platform.observability.logging import sanitize_log_fields


def test_trace_context_propagates_task_and_correlation_then_resets():
    original = get_trace_context()
    context = TraceContext.from_identity(
        task_id="root-1", correlation_id="hop-1", agent_id="data_query",
        tenant_id="tenant-raw", user_id="user-raw",
    )
    with trace_context(context):
        current = get_trace_context()
        assert current.task_id == "root-1" and current.correlation_id == "hop-1"
        assert current.tenant_hash == identity_hash("tenant-raw")
        assert current.user_hash == identity_hash("user-raw")
        assert "tenant-raw" not in repr(current) and "user-raw" not in repr(current)
    assert get_trace_context() == original


def test_json_formatter_injects_context_and_redacts_message_and_fields():
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1,
        "authorization=Bearer raw.jwt.token password=hunter2", (), None,
    )
    record.event = "safe_event"
    record.query_hash = "abc123"
    record.prompt_tokens = 10
    record.api_key = "real-key"
    with trace_context(TraceContext.from_identity(
        task_id="root", correlation_id="hop", tenant_id="tenant-secret", user_id="user-secret"
    )):
        output = json.loads(JsonFormatter().format(record))
    encoded = json.dumps(output)
    assert output["task_id"] == "root" and output["correlation_id"] == "hop"
    assert output["query_hash"] == "abc123" and output["prompt_tokens"] == 10
    assert "raw.jwt.token" not in encoded and "hunter2" not in encoded
    assert "real-key" not in encoded and "tenant-secret" not in encoded


def test_sensitive_structured_fields_are_centrally_redacted():
    safe = sanitize_log_fields({
        "password": "p", "token": "t", "payload": {"order": 1},
        "query": "refund", "error_type": "ValueError",
    })
    assert safe["password"] == safe["token"] == safe["query"] == "[REDACTED]"
    assert safe["payload"] == "[REDACTED]"
    assert safe["error_type"] == "ValueError"


def test_request_performance_joins_agent_tasks_by_root_task_id_and_cleans_up():
    begin_request_performance("root-a")
    begin_request_performance("root-b")
    try:
        with trace_context(TraceContext(task_id="root-a", agent_id="data_query")):
            record_llm_usage(7, 3, 10, 0.001)
            assert get_performance_summary()["total_tokens"] == 10
        with trace_context(TraceContext(task_id="root-b", agent_id="knowledge_rag")):
            assert get_performance_summary()["total_tokens"] == 0
            record_llm_usage(2, 1, 3, None)
        with trace_context(TraceContext(task_id="root-a")):
            assert get_performance_summary()["llm_calls"] == 1
        with trace_context(TraceContext(task_id="root-b")):
            assert get_performance_summary()["total_tokens"] == 3
    finally:
        finish_request_performance("root-a")
        finish_request_performance("root-b")
