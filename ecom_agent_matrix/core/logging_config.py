"""结构化日志配置。"""
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any

from ecom_agent_matrix.platform.observability.context import get_trace_context
from ecom_agent_matrix.platform.observability.logging import (
    sanitize_log_fields,
    sanitize_message,
)


class JsonFormatter(logging.Formatter):
    """输出 JSON 行日志，便于 ELK / Loki 采集。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_message(record.getMessage()),
        }
        trace = get_trace_context()
        trace_fields = {
            "task_id": trace.task_id,
            "correlation_id": trace.correlation_id,
            "agent": trace.agent_id,
            "workflow": trace.workflow,
            "skill": trace.skill_name,
            "tenant_hash": trace.tenant_hash,
            "user_hash": trace.user_hash,
        }
        payload.update({key: value for key, value in trace_fields.items() if value})
        # 业务字段通过 logger.info(..., extra={...}) 传入
        for key in (
            "task_id",
            "correlation_id",
            "query_hash",
            "query_length",
            "lang",
            "recall_count",
            "latency_ms",
            "cached",
            "agent",
            "event",
            "model",
            "mode",
            "requested",
            "applied",
            "error_type",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "has_reasoning",
            "attempt",
            "max_retries",
            "delay_s",
            "status",
            "provider",
            "method",
            "route",
            "status_class",
            "workflow",
            "skill",
            "error_code",
            "component",
            "estimated_cost_usd",
        ):
            if hasattr(record, key):
                payload.update(sanitize_log_fields({key: getattr(record, key)}))
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
            payload["stack"] = [
                f"{frame.filename}:{frame.lineno}:{frame.name}"
                for frame in traceback.extract_tb(record.exc_info[2])[-12:]
            ]
        return json.dumps(payload, ensure_ascii=False)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
