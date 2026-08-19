from __future__ import annotations

import time
from types import SimpleNamespace

import jwt
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from ecom_agent_matrix.api.auth import (
    authenticate_api_key,
    authenticate_jwt,
    validate_security_configuration,
)
from ecom_agent_matrix.api.schemas import TaskCreateRequest
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.security.errors import SecurityConfigurationError


def _jwt(**updates) -> str:
    claims = {
        "sub": "subject-1",
        "user_id": "user-1",
        "tenant_id": "tenant-1",
        "store_id": "store-1",
        "roles": ["viewer"],
        "scope": "knowledge:read commerce:read",
        "iss": "issuer-1",
        "aud": "audience-1",
        "exp": int(time.time()) + 300,
    }
    claims.update(updates)
    return jwt.encode(claims, "jwt-secret", algorithm="HS256")


def test_valid_and_invalid_api_key(monkeypatch):
    monkeypatch.setattr(settings, "APP_ENV", "development")
    monkeypatch.setattr(settings, "API_KEY", "dev-key")
    monkeypatch.setattr(settings, "DEV_ROLES", "viewer")
    security = authenticate_api_key("dev-key")
    assert security.auth_type == "api_key" and security.authenticated
    assert security.tenant_id == settings.DEV_TENANT_ID
    with pytest.raises(HTTPException) as exc:
        authenticate_api_key("wrong")
    assert exc.value.status_code == 401


def test_valid_jwt_becomes_normalized_context_without_raw_token(monkeypatch):
    monkeypatch.setattr(settings, "JWT_SECRET", "jwt-secret")
    monkeypatch.setattr(settings, "JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(settings, "JWT_ISSUER", "issuer-1")
    monkeypatch.setattr(settings, "JWT_AUDIENCE", "audience-1")
    token = _jwt()
    security = authenticate_jwt(token)
    assert security.auth_type == "jwt" and security.user_id == "user-1"
    assert security.scopes == frozenset({"knowledge:read", "commerce:read"})
    assert token not in str(security.model_dump())


@pytest.mark.parametrize(
    "claims",
    [
        {"exp": int(time.time()) - 1},
        {"iss": "wrong"},
        {"aud": "wrong"},
        {"tenant_id": ""},
        {"user_id": ""},
    ],
)
def test_invalid_jwt_claims_return_401(monkeypatch, claims):
    monkeypatch.setattr(settings, "JWT_SECRET", "jwt-secret")
    monkeypatch.setattr(settings, "JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(settings, "JWT_ISSUER", "issuer-1")
    monkeypatch.setattr(settings, "JWT_AUDIENCE", "audience-1")
    with pytest.raises(HTTPException) as exc:
        authenticate_jwt(_jwt(**claims))
    assert exc.value.status_code == 401
    assert "jwt-secret" not in str(exc.value.detail)


@pytest.mark.parametrize("field", ["tenant_id", "user_id", "store_id", "roles", "scopes"])
def test_task_payload_rejects_identity_spoof_fields(field):
    with pytest.raises(ValidationError) as exc:
        TaskCreateRequest(query="q", payload={field: "fake"})
    assert "SECURITY_FIELD_FORBIDDEN" in str(exc.value)


def test_production_security_config_is_fail_closed():
    base = dict(
        APP_ENV="production", AUTH_MODE="jwt", ALLOW_INSECURE_LOCAL=False,
        JWT_SECRET="", JWT_ISSUER="issuer", JWT_AUDIENCE="audience",
        JWT_ALGORITHM="HS256",
    )
    with pytest.raises(SecurityConfigurationError):
        validate_security_configuration(SimpleNamespace(**base))
    with pytest.raises(SecurityConfigurationError):
        validate_security_configuration(SimpleNamespace(**{
            **base, "AUTH_MODE": "api_key", "ALLOW_INSECURE_LOCAL": True,
        }))
    validate_security_configuration(SimpleNamespace(**{
        **base, "JWT_SECRET": "configured-secret",
    }))

