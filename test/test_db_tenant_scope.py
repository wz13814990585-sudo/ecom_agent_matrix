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
    def __init__(self, rows=None, *, fail_on_sql=""):
        self.calls = []
        self.rows = list(rows or [(1,)])
        self.description = None
        self.fail_on_sql = fail_on_sql
        self.transactions = 0

    def begin(self):
        self.transactions += 1
        cursor = self

        class _Transaction:
            async def __aenter__(self):
                cursor.calls.append(("BEGIN", []))
                return self

            async def __aexit__(self, exc_type, _exc, _tb):
                cursor.calls.append(("ROLLBACK" if exc_type else "COMMIT", []))
                return False

        return _Transaction()

    async def execute(self, sql, params=None):
        self.calls.append((sql, list(params or [])))
        if self.fail_on_sql and self.fail_on_sql in sql:
            raise RuntimeError("simulated database failure")
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
    assert sqls.count("BEGIN") == 2
    assert sqls.count("COMMIT") == 2
    assert sqls.count("SET TRANSACTION READ ONLY") == 2
    assert "SET LOCAL TRANSACTION READ ONLY" not in sqls
    scope_calls = [params for sql, params in cursor.calls if "set_config('app.tenant_id'" in sql]
    assert scope_calls == [["tenant-a", "store-a"], ["tenant-b", "store-b"]]
    assert cursor.transactions == 2

    first = sqls[: sqls.index("COMMIT") + 1]
    assert first.index("BEGIN") < first.index("SET TRANSACTION READ ONLY")
    assert first.index("SET TRANSACTION READ ONLY") < next(
        i for i, sql in enumerate(first) if "set_config('statement_timeout'" in sql
    )
    tenant_scope_at = next(
        i for i, sql in enumerate(first) if "set_config('app.tenant_id'" in sql
    )
    assert tenant_scope_at < first.index("SELECT 1") < first.index("COMMIT")


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
    sqls = [sql for sql, _ in write_cursor.calls]
    assert sqls[0] == "BEGIN" and sqls[-1] == "COMMIT"
    assert "SET TRANSACTION READ ONLY" not in sqls
    timeout_at = next(i for i, sql in enumerate(sqls) if "set_config('statement_timeout'" in sql)
    scope_at = next(i for i, sql in enumerate(sqls) if "set_config('app.tenant_id'" in sql)
    mutation_at = next(i for i, sql in enumerate(sqls) if sql.startswith("INSERT"))
    assert 0 < timeout_at < scope_at < mutation_at < len(sqls) - 1
    assert ["tenant-a", "store-a"] in [params for _, params in write_cursor.calls]
    assert write_cursor.transactions == 1


def test_bounded_read_fetches_one_extra_and_marks_truncated():
    cursor = _Cursor([(1,), (2,), (3,)])
    pool = _Pool(cursor)

    async def scenario():
        with patch.object(AsyncPGClient, "get_read_pool", new=AsyncMock(return_value=pool)):
            return await AsyncPGClient.execute_read_bounded(
                "SELECT id FROM ecom_order", scope=_scope("t", "s"), max_rows=2
            )

    assert asyncio.run(scenario()) == ([(1,), (2,)], True)


def test_scoped_business_exception_rolls_back_and_connection_can_be_reused():
    cursor = _Cursor(fail_on_sql="BROKEN")
    pool = _Pool(cursor)

    async def scenario():
        with patch.object(AsyncPGClient, "get_write_pool", new=AsyncMock(return_value=pool)):
            try:
                await AsyncPGClient.execute_write(
                    "UPDATE BROKEN SET value=1", scope=_scope("tenant-a", "store-a")
                )
            except RuntimeError:
                pass
            cursor.fail_on_sql = ""
            await AsyncPGClient.execute_write(
                "UPDATE healthy SET value=1", scope=_scope("tenant-b", "store-b")
            )

    asyncio.run(scenario())
    sqls = [sql for sql, _ in cursor.calls]
    assert sqls.count("BEGIN") == 2
    assert sqls.count("ROLLBACK") == 1
    assert sqls.count("COMMIT") == 1
    scopes = [params for sql, params in cursor.calls if "set_config('app.tenant_id'" in sql]
    assert scopes == [["tenant-a", "store-a"], ["tenant-b", "store-b"]]


def test_realistic_aiopg_fake_exposes_transaction_only_on_cursor():
    cursor = _Cursor()
    connection = _Connection(cursor)
    assert not hasattr(connection, "begin")
    assert callable(cursor.begin)
