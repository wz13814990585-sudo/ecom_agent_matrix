from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.core.security import TenantScope
from ecom_agent_matrix.db.base import AsyncPGClient


class _CM:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_args):
        return False


class _Cursor:
    def __init__(self, rows=None):
        self.calls = []
        self.rows = list(rows or [(1,)])
        self.description = None

    async def execute(self, sql, params=None):
        self.calls.append((sql, list(params or [])))
        upper = sql.strip().upper()
        self.description = object() if upper.startswith("SELECT") or "RETURNING" in upper else None

    async def fetchall(self):
        return self.rows

    async def fetchmany(self, size):
        return self.rows[:size]

    async def fetchone(self):
        return self.rows[0] if self.rows else None


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.transactions = 0

    def begin(self):
        self.transactions += 1
        return _CM(self)

    def cursor(self):
        return _CM(self._cursor)


class _Pool:
    def __init__(self, cursor):
        self.connection = _Connection(cursor)

    def acquire(self):
        return _CM(self.connection)


def _scope(tenant, store):
    return TenantScope(tenant_id=tenant, store_id=store, identity_trusted=True)


def test_read_pool_transaction_is_read_only_and_scope_is_transaction_local():
    cursor = _Cursor([(1,)])
    pool = _Pool(cursor)

    async def scenario():
        with patch.object(AsyncPGClient, "get_read_pool", new=AsyncMock(return_value=pool)):
            first = await AsyncPGClient.execute_read("SELECT 1", scope=_scope("tenant-a", "store-a"))
            second = await AsyncPGClient.execute_read("SELECT 1", scope=_scope("tenant-b", "store-b"))
        return first, second

    assert asyncio.run(scenario()) == ([(1,)], [(1,)])
    sqls = [sql for sql, _ in cursor.calls]
    assert sqls.count("SET LOCAL TRANSACTION READ ONLY") == 2
    scope_calls = [params for sql, params in cursor.calls if "set_config('app.tenant_id'" in sql]
    assert scope_calls == [["tenant-a", "store-a"], ["tenant-b", "store-b"]]
    assert pool.connection.transactions == 2


def test_write_path_uses_write_pool_and_not_read_pool():
    write_cursor = _Cursor([(7,)])
    write_pool = _Pool(write_cursor)
    read = AsyncMock()

    async def scenario():
        with patch.object(AsyncPGClient, "get_write_pool", new=AsyncMock(return_value=write_pool)), patch.object(
            AsyncPGClient, "get_read_pool", new=read
        ):
            return await AsyncPGClient.execute_write(
                "INSERT INTO risk_record(order_no) VALUES (%s) RETURNING id",
                ["ORD-1"],
                scope=_scope("tenant-a", "store-a"),
            )

    assert asyncio.run(scenario()) == [(7,)]
    read.assert_not_awaited()
    assert not any(sql == "SET LOCAL TRANSACTION READ ONLY" for sql, _ in write_cursor.calls)
    assert ["tenant-a", "store-a"] in [params for _, params in write_cursor.calls]


def test_bounded_read_fetches_one_extra_and_marks_truncated():
    cursor = _Cursor([(1,), (2,), (3,)])
    pool = _Pool(cursor)

    async def scenario():
        with patch.object(AsyncPGClient, "get_read_pool", new=AsyncMock(return_value=pool)):
            return await AsyncPGClient.execute_read_bounded(
                "SELECT id FROM ecom_order", scope=_scope("t", "s"), max_rows=2
            )

    assert asyncio.run(scenario()) == ([(1,), (2,)], True)

