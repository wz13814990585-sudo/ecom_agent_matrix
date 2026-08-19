"""safe_sql_query / NL 白名单模板。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.modules.skills.sql_tool import nl_to_readonly_sql, sanitize_readonly_sql


def test_nl_to_sql_order_count():
    sql, label, err = nl_to_readonly_sql("数据库里有多少订单")
    assert err == ""
    assert "ecom_order" in sql
    assert label == "订单总数"


def test_nl_to_sql_tables():
    sql, label, err = nl_to_readonly_sql("查询数据库有哪些表")
    assert err == ""
    assert "information_schema.tables" in sql


def test_sanitize_blocks_write():
    cleaned, err = sanitize_readonly_sql("DELETE FROM ecom_goods")
    assert cleaned is None
    assert "仅允许" in err or "禁止" in err
