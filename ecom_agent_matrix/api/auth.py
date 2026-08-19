"""HTTP resource-server authentication producing a trusted SecurityContext."""
from __future__ import annotations

import secrets
from collections.abc import Iterable
from typing import Any

import jwt
from fastapi import Header, HTTPException, status

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.security import SecurityConfigurationError, SecurityContext


def _items(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        values: Iterable[Any] = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        values = ()
    return frozenset(str(item).strip() for item in values if str(item).strip())


def validate_security_configuration(config=settings) -> None:
    env = str(config.APP_ENV or "").strip().lower()
    mode = str(config.AUTH_MODE or "").strip().lower()
    if mode not in {"api_key", "jwt"}:
        raise SecurityConfigurationError("AUTH_MODE must be api_key or jwt")
    if env == "production":
        if mode != "jwt":
            raise SecurityConfigurationError("production requires AUTH_MODE=jwt")
        if bool(config.ALLOW_INSECURE_LOCAL):
            raise SecurityConfigurationError("production cannot allow insecure local auth")
        if any(
            not str(value or "").strip()
            for value in (config.JWT_SECRET, config.JWT_ISSUER, config.JWT_AUDIENCE)
        ):
            raise SecurityConfigurationError(
                "production JWT_SECRET, JWT_ISSUER and JWT_AUDIENCE are required"
            )
    if mode == "jwt":
        if str(config.JWT_ALGORITHM or "").strip() != "HS256":
            raise SecurityConfigurationError("Phase 5A supports JWT_ALGORITHM=HS256 only")
        if any(
            not str(value or "").strip()
            for value in (config.JWT_SECRET, config.JWT_ISSUER, config.JWT_AUDIENCE)
        ):
            raise SecurityConfigurationError(
                "JWT_SECRET, JWT_ISSUER and JWT_AUDIENCE are required in jwt mode"
            )


def _dev_principal(*, authenticated: bool, auth_type: str) -> SecurityContext:
    return SecurityContext(
        subject=str(settings.DEV_USER_ID).strip(),
        user_id=str(settings.DEV_USER_ID).strip(),
        tenant_id=str(settings.DEV_TENANT_ID).strip(),
        store_id=str(settings.DEV_STORE_ID).strip(),
        roles=_items(settings.DEV_ROLES),
        scopes=_items(settings.DEV_SCOPES),
        auth_type=auth_type,
        authenticated=authenticated,
    )


def authenticate_api_key(api_key: str | None) -> SecurityContext:
    expected = str(settings.API_KEY or "").strip()
    if expected and api_key and secrets.compare_digest(api_key.strip(), expected):
        return _dev_principal(authenticated=True, auth_type="api_key")
    env = str(settings.APP_ENV or "").strip().lower()
    if not expected and env in {"development", "test"} and settings.ALLOW_INSECURE_LOCAL:
        # Explicit local-only trust boundary; production configuration forbids it.
        return _dev_principal(authenticated=True, auth_type="system")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )


def authenticate_jwt(token: str | None) -> SecurityContext:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
        )
    try:
        claims = jwt.decode(
            token,
            str(settings.JWT_SECRET),
            algorithms=[str(settings.JWT_ALGORITHM)],
            issuer=str(settings.JWT_ISSUER),
            audience=str(settings.JWT_AUDIENCE),
            options={"require": ["exp", "sub"]},
        )
        subject = str(claims.get("sub") or "").strip()
        user_id = str(claims.get("user_id") or "").strip()
        tenant_id = str(claims.get("tenant_id") or "").strip()
        store_id = str(claims.get("store_id") or "").strip()
        if not subject or not user_id or not tenant_id or not store_id:
            raise ValueError("required identity claim missing")
        return SecurityContext(
            subject=subject,
            user_id=user_id,
            tenant_id=tenant_id,
            store_id=store_id,
            roles=_items(claims.get("roles") or claims.get("role")),
            scopes=_items(claims.get("scopes") or claims.get("scope")),
            auth_type="jwt",
            authenticated=True,
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired bearer token",
        ) from None


async def get_current_security_context(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> SecurityContext:
    mode = str(settings.AUTH_MODE or "").strip().lower()
    if mode == "api_key":
        return authenticate_api_key(x_api_key)
    if mode == "jwt":
        prefix, _, token = str(authorization or "").partition(" ")
        return authenticate_jwt(token.strip() if prefix.lower() == "bearer" else "")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication is not configured",
    )


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> SecurityContext:
    """Legacy dependency name retained for external imports."""
    return authenticate_api_key(x_api_key)


__all__ = [
    "authenticate_api_key",
    "authenticate_jwt",
    "get_current_security_context",
    "require_api_key",
    "validate_security_configuration",
]
