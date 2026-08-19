from pathlib import Path


DB_DIR = Path(__file__).resolve().parents[1] / "ecom_agent_matrix" / "db"
MIGRATION = DB_DIR / "migrations" / "20260819_01_tenant_isolation.sql"


def _normalized(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").lower().split())


def test_migration_and_fresh_schemas_cover_all_tenant_tables():
    migration = _normalized(MIGRATION)
    fresh = _normalized(DB_DIR / "business_tables.sql") + " " + _normalized(DB_DIR / "vector_tables.sql")
    tables = {
        "ecom_goods", "ecom_order", "competitor_price", "risk_record",
        "vector_goods_kb", "agent_long_memory", "finetune_dataset", "mcp_message_log",
        "security_approval", "security_audit_log",
    }
    for table in tables:
        assert table in migration
        assert table in fresh
    assert migration.count("alter column tenant_id set not null") >= 8
    assert migration.count("alter column store_id set not null") >= 8


def test_goods_and_order_unique_keys_are_tenant_store_scoped():
    migration = _normalized(MIGRATION)
    fresh = _normalized(DB_DIR / "business_tables.sql")
    key = "unique (tenant_id, store_id, sku)"
    order_key = "unique (tenant_id, store_id, order_no)"
    assert key in migration and key in fresh
    assert order_key in migration and order_key in fresh
    assert "drop constraint if exists ecom_goods_sku_key" in migration
    assert "drop constraint if exists ecom_order_order_no_key" in migration


def test_rls_is_forced_and_policy_checks_both_scope_dimensions_for_read_and_write():
    for path in (MIGRATION, DB_DIR / "business_tables.sql", DB_DIR / "vector_tables.sql"):
        sql = _normalized(path)
        assert "enable row level security" in sql
        assert "force row level security" in sql
        assert "using (tenant_id = current_setting(''app.tenant_id'', true) and store_id = current_setting(''app.store_id'', true))" in sql
        assert "with check (tenant_id = current_setting(''app.tenant_id'', true) and store_id = current_setting(''app.store_id'', true))" in sql


def test_security_role_example_contains_no_password_and_read_role_is_select_only():
    sql = _normalized(DB_DIR / "security_roles.sql.example")
    assert "password '" not in sql
    assert "grant select on" in sql
    assert "app_read_role" in sql and "app_write_role" in sql
    assert "nosuperuser nobypassrls" in sql
