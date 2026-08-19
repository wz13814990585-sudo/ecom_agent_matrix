"""Verified identity models propagated through internal envelopes."""
from __future__ import annotations

from typing import Literal
import hashlib

from pydantic import BaseModel, ConfigDict, Field


class SecurityContext(BaseModel):
    """Normalized verified claims only; never contains credentials or raw tokens."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    subject: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    store_id: str = Field(min_length=1)
    roles: frozenset[str] = Field(default_factory=frozenset)
    scopes: frozenset[str] = Field(default_factory=frozenset)
    auth_type: Literal["jwt", "api_key", "system"]
    authenticated: bool


def security_log_fields(security: SecurityContext | None) -> dict[str, object]:
    """Return bounded audit fields without raw principal identifiers or credentials."""
    if security is None:
        return {"auth_type": "legacy", "identity_trusted": False}

    def digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    return {
        "tenant_hash": digest(security.tenant_id),
        "user_hash": digest(security.user_id),
        "auth_type": security.auth_type,
        "roles": sorted(security.roles),
        "identity_trusted": security.authenticated,
    }


__all__ = ["SecurityContext", "security_log_fields"]
