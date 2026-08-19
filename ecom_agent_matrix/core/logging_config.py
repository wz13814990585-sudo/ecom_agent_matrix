"""结构化日志配置。"""
import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """输出 JSON 行日志，便于 ELK / Loki 采集。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 业务字段通过 logger.info(..., extra={...}) 传入
        for key in (
            "task_id",
            "query",
            "lang",
            "recall_count",
            "latency_ms",
            "cached",
            "agent",
            "event",
            "error",
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
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
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
