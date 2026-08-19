"""Parser 与 SQL Skill 共享的只读 SQL 白名单策略。"""
from __future__ import annotations

import re

from ecom_agent_matrix.config.constants import (
    TABLE_COMPETITOR,
    TABLE_GOODS,
    TABLE_ORDER,
    TABLE_RISK_LOG,
)

_FORBIDDEN = re.compile(
    r"\b("
    r"delete|drop|update|alter|truncate|insert|merge|replace|copy|"
    r"create|grant|revoke|call|do|execute|exec|into|"
    r"set|vacuum|analyze|reindex|cluster|comment|lock|notify|listen|"
    r"pg_sleep|lo_import|lo_export"
    r")\b",
    re.IGNORECASE,
)
_COMMENT_LINE = re.compile(r"--[^\n]*")
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)

NL_SQL_TEMPLATES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"(订单|order).*(有多少|数量|总数|count)|有多少.*订单", re.I),
        f"SELECT COUNT(*) AS cnt FROM {TABLE_ORDER}",
        "订单总数",
    ),
    (
        re.compile(r"(商品|goods|sku).*(有多少|数量|总数|count)|有多少.*商品", re.I),
        f"SELECT COUNT(*) AS cnt FROM {TABLE_GOODS}",
        "商品总数",
    ),
    (
        re.compile(r"按店铺.*(统计|分组|数量)|店铺.*商品数", re.I),
        (
            f"SELECT store_id, store_name, COUNT(*) AS cnt FROM {TABLE_GOODS} "
            "GROUP BY store_id, store_name ORDER BY cnt DESC"
        ),
        "按店铺统计商品数",
    ),
    (
        re.compile(r"竞品价|competitor_price|报价记录", re.I),
        (
            f"SELECT target_sku, competitor_name, compete_price, crawl_time "
            f"FROM {TABLE_COMPETITOR} ORDER BY id DESC LIMIT 20"
        ),
        "最近竞品报价",
    ),
    (
        re.compile(r"风控|risk_record|风险记录", re.I),
        f"SELECT * FROM {TABLE_RISK_LOG} ORDER BY id DESC LIMIT 20",
        "最近风控记录",
    ),
    (
        re.compile(r"有哪些表|list\s+tables|数据库.*表", re.I),
        (
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        ),
        "当前库表清单",
    ),
)


def sanitize_readonly_sql(sql: str) -> tuple[str | None, str]:
    """校验并规范化单条 SELECT/WITH…SELECT。"""
    if not sql or not str(sql).strip():
        return None, "缺少 sql 查询语句参数"
    cleaned = _COMMENT_BLOCK.sub(" ", str(sql))
    cleaned = _COMMENT_LINE.sub(" ", cleaned).strip()
    if ";" in cleaned.rstrip(";"):
        return None, "禁止多语句 SQL（仅允许单条 SELECT/WITH）"
    cleaned = cleaned.rstrip(";").strip()
    if not cleaned:
        return None, "SQL 为空"
    head = cleaned.lstrip().lower()
    if not (head.startswith("select") or head.startswith("with")):
        return None, "仅允许执行 SELECT / WITH…SELECT 只读查询"
    if head.startswith("with") and not re.search(r"\bselect\b", cleaned, re.IGNORECASE):
        return None, "WITH 语句必须包含 SELECT"
    hit = _FORBIDDEN.search(cleaned)
    if hit:
        return None, f"禁止高危 SQL 关键字：{hit.group(1)}"
    return cleaned, ""


def nl_to_readonly_sql(text: str) -> tuple[str | None, str, str]:
    """将受支持的自然语言查询映射为固定只读 SQL 模板。"""
    query = str(text or "").strip()
    if not query:
        return None, "", "缺少自然语言查询"
    if re.match(r"^\s*(select|with)\b", query, re.I):
        cleaned, error = sanitize_readonly_sql(query)
        return cleaned, "raw_sql", error
    for pattern, sql, label in NL_SQL_TEMPLATES:
        if pattern.search(query):
            return sql, label, ""
    return (
        None,
        "",
        "暂不支持该自然语言查库；可试：有多少订单/商品、按店铺统计、竞品报价、有哪些表；"
        "或传 sql=SELECT ...",
    )
