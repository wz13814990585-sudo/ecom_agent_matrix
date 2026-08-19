"""Trusted identity and authorization boundary."""

from ecom_agent_matrix.core.security.context import SecurityContext, security_log_fields
from ecom_agent_matrix.core.security.errors import (
    AuthenticationError,
    AuthorizationError,
    SecurityConfigurationError,
)
from ecom_agent_matrix.core.security.policy import (
    authorize_task,
    authorize_task_types,
    effective_scopes,
    is_task_authorized,
    required_scopes_for_task,
    require_trusted_ingress,
)

__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "SecurityConfigurationError",
    "SecurityContext",
    "authorize_task",
    "authorize_task_types",
    "effective_scopes",
    "is_task_authorized",
    "required_scopes_for_task",
    "security_log_fields",
    "require_trusted_ingress",
]
