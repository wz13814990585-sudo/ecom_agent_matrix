"""Best-effort tenant-scoped security audit events with bounded metadata."""
from __future__ import annotations

import json
from typing import Any

from ecom_agent_matrix.core.security.scope import TenantScope
from ecom_agent_matrix.db.base import AsyncPGClient

AUDIT_EVENTS = frozenset({
    "AUTHORIZATION_DENIED", "APPROVAL_REQUESTED", "APPROVAL_APPROVED",
    "APPROVAL_REJECTED", "APPROVAL_CONSUMED", "HIGH_RISK_EXECUTION_STARTED",
    "HIGH_RISK_EXECUTION_SUCCEEDED", "HIGH_RISK_EXECUTION_FAILED",
})


def _safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in list((metadata or {}).items())[:20]:
        name = str(key)[:64]
        if any(marker in name.lower() for marker in ("token", "secret", "password", "sql", "prompt", "payload")):
            continue
        if isinstance(value, (bool, int, float)) or value is None:
            safe[name] = value
        elif isinstance(value, str):
            safe[name] = value[:200]
    return safe


async def record_audit_event(
    event_type: str,
    *,
    scope: TenantScope,
    task_id: str = "",
    user_id: str = "",
    agent_id: str = "",
    skill_name: str = "",
    approval_id: str = "",
    outcome: str = "",
    reason_code: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    if event_type not in AUDIT_EVENTS or not scope.usable:
        return
    try:
        await AsyncPGClient.execute_write(
            """
            INSERT INTO security_audit_log(
              event_type, task_id, tenant_id, store_id, user_id, agent_id,
              skill_name, approval_id, outcome, reason_code, metadata_json
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,NULLIF(%s,'')::uuid,%s,%s,%s::jsonb)
            """,
            [event_type, task_id, scope.tenant_id, scope.store_id, user_id, agent_id,
             skill_name, approval_id, outcome, reason_code,
             json.dumps(_safe_metadata(metadata), ensure_ascii=False)],
            scope=scope,
        )
    except Exception:
        # Audit storage is best effort here; callers never expose DB exceptions.
        return


__all__ = ["AUDIT_EVENTS", "record_audit_event"]
