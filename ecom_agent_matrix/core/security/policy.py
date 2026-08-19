"""Central task-level RBAC policy, independent from planners and payload params."""
from __future__ import annotations

from collections.abc import Iterable

from ecom_agent_matrix.core.security.context import SecurityContext
from ecom_agent_matrix.core.security.errors import AuthorizationError
from ecom_agent_matrix.core.security.errors import AuthenticationError

KNOWLEDGE_READ = "knowledge:read"
COMMERCE_READ = "commerce:read"
OPERATIONS_EXECUTE = "operations:execute"
RISK_WRITE = "risk:write"
RISK_APPROVE = "risk:approve"
SYSTEM_READ = "system:read"

ALL_SCOPES = frozenset(
    {KNOWLEDGE_READ, COMMERCE_READ, OPERATIONS_EXECUTE, RISK_WRITE, RISK_APPROVE, SYSTEM_READ}
)
ROLE_SCOPES: dict[str, frozenset[str]] = {
    "viewer": frozenset({KNOWLEDGE_READ, COMMERCE_READ}),
    "operator": frozenset({KNOWLEDGE_READ, COMMERCE_READ, OPERATIONS_EXECUTE}),
    "risk_operator": frozenset({COMMERCE_READ, RISK_WRITE}),
    "risk_approver": frozenset({COMMERCE_READ, RISK_APPROVE}),
    "admin": ALL_SCOPES,
}

TASK_REQUIRED_SCOPES: dict[str, frozenset[str]] = {
    "knowledge_qa": frozenset({KNOWLEDGE_READ}),
    "goods_search": frozenset({COMMERCE_READ}),
    "goods_catalog": frozenset({COMMERCE_READ}),
    "stock_analysis": frozenset({COMMERCE_READ}),
    "competitor_watch": frozenset({COMMERCE_READ}),
    "order_query": frozenset({COMMERCE_READ}),
    "ad_query": frozenset({COMMERCE_READ}),
    "social_marketing": frozenset({OPERATIONS_EXECUTE}),
    "customer_service": frozenset({OPERATIONS_EXECUTE}),
    "ad_optimize": frozenset({OPERATIONS_EXECUTE}),
    "ops_report": frozenset({OPERATIONS_EXECUTE}),
    "risk_control": frozenset({RISK_WRITE}),
    "data_check": frozenset({SYSTEM_READ}),
}


def effective_scopes(security: SecurityContext) -> frozenset[str]:
    scopes = set(security.scopes)
    for role in security.roles:
        scopes.update(ROLE_SCOPES.get(role, ()))
    return frozenset(scopes)


def required_scopes_for_task(task_type: str) -> frozenset[str]:
    return TASK_REQUIRED_SCOPES.get(str(task_type or "").strip(), frozenset())


def is_task_authorized(security: SecurityContext | None, task_type: str) -> bool:
    required = required_scopes_for_task(task_type)
    if security is None or not security.authenticated or not required:
        return False
    return required.issubset(effective_scopes(security))


def authorize_task(security: SecurityContext | None, task_type: str) -> None:
    if not is_task_authorized(security, task_type):
        raise AuthorizationError(task_type)


def authorize_task_types(
    security: SecurityContext | None,
    task_types: Iterable[str],
) -> None:
    for task_type in task_types:
        authorize_task(security, task_type)


def require_trusted_ingress(
    security: SecurityContext | None,
    *,
    app_env: str,
) -> None:
    """Production MCP ingress is fail-closed; legacy direct calls remain compatible."""
    if str(app_env or "").strip().lower() == "production" and (
        security is None or not security.authenticated
    ):
        raise AuthenticationError("Trusted SecurityContext is required")


__all__ = [
    "ALL_SCOPES",
    "ROLE_SCOPES",
    "TASK_REQUIRED_SCOPES",
    "authorize_task",
    "authorize_task_types",
    "effective_scopes",
    "is_task_authorized",
    "required_scopes_for_task",
    "require_trusted_ingress",
]
