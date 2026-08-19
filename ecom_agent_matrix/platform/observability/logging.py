"""Central structured-log redaction."""
from __future__ import annotations

import re
from typing import Any

SENSITIVE_MARKERS = (
    "token", "secret", "password", "api_key", "authorization", "cookie",
    "session_key", "jwt", "payload", "prompt", "query",
)
_BEARER = re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+")
_ASSIGNMENT = re.compile(
    r"(?i)(password|secret|api[_-]?key|token|authorization|cookie|session[_-]?key|jwt)"
    r"\s*[:=]\s*[^\s,;]+"
)


def sanitize_message(message: str) -> str:
    text = _BEARER.sub("Bearer [REDACTED]", str(message or ""))
    return _ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)


def sanitize_log_fields(fields: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in fields.items():
        lowered = str(key).lower()
        explicitly_safe = lowered in {
            "query_hash", "query_length", "prompt_tokens", "completion_tokens", "total_tokens"
        }
        if not explicitly_safe and any(marker in lowered for marker in SENSITIVE_MARKERS):
            safe[key] = "[REDACTED]"
        elif isinstance(value, str):
            safe[key] = sanitize_message(value)[:1000]
        elif isinstance(value, (bool, int, float)) or value is None:
            safe[key] = value
    return safe


__all__ = ["sanitize_log_fields", "sanitize_message"]
