from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ecom_agent_matrix.db.base import (
    AsyncPGClient,
    validate_database_runtime_roles,
    validate_database_security_configuration,
)
from test_db_tenant_scope import _Cursor, _Pool


def _config(**updates):
    values = {
        "APP_ENV": "production",
        "PG_READ_USER": "app_read",
        "PG_READ_PWD": "read-secret",
        "PG_WRITE_USER": "app_write",
        "PG_WRITE_PWD": "write-secret",
    }
    values.update(updates)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "updates",
    [
        {"PG_READ_USER": ""},
        {"PG_READ_PWD": ""},
        {"PG_WRITE_USER": ""},
        {"PG_WRITE_PWD": ""},
        {"PG_READ_USER": "same", "PG_WRITE_USER": "same"},
    ],
)
def test_production_requires_distinct_read_write_credentials(updates):
    with pytest.raises(RuntimeError, match="READ_DB_CONFIGURATION_INVALID"):
        validate_database_security_configuration(_config(**updates))


def test_development_retains_legacy_database_compatibility():
    validate_database_security_configuration(_config(APP_ENV="development", PG_READ_USER="", PG_WRITE_USER=""))


@pytest.mark.parametrize("role_row", [[(True, False)], [(False, True)]])
def test_production_rejects_superuser_or_bypassrls_runtime_role(role_row):
    pool = _Pool(_Cursor(role_row))

    async def scenario():
        with patch.object(AsyncPGClient, "get_read_pool", new=AsyncMock(return_value=pool)), patch.object(
            AsyncPGClient, "get_write_pool", new=AsyncMock(return_value=pool)
        ):
            await validate_database_runtime_roles(_config())

    with pytest.raises(RuntimeError, match="READ_DB_ROLE_INVALID"):
        asyncio.run(scenario())


def test_production_accepts_non_privileged_distinct_runtime_roles():
    read_pool = _Pool(_Cursor([(False, False)]))
    write_pool = _Pool(_Cursor([(False, False)]))

    async def scenario():
        with patch.object(AsyncPGClient, "get_read_pool", new=AsyncMock(return_value=read_pool)), patch.object(
            AsyncPGClient, "get_write_pool", new=AsyncMock(return_value=write_pool)
        ):
            await validate_database_runtime_roles(_config())

    asyncio.run(scenario())
