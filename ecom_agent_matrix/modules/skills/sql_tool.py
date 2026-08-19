"""安全只读 SQL 查询工具（由 data_check Agent 按需调用：payload.sql / custom_sql / NL）。"""
from __future__ import annotations

import re

from ecom_agent_matrix.config.constants import (
    TABLE_COMPETITOR,
    TABLE_GOODS,
    TABLE_ORDER,
    TABLE_RISK_LOG,
)
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill
from ecom_agent_matrix.db.base import AsyncPGClient

# 整词匹配，避免误伤列名；覆盖常见写操作与旁路
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

# 常见自然语言 → 白名单 SQL 模板（禁止任意 NL2SQL 直连写库）
_NL_SQL_TEMPLATES: list[tuple[re.Pattern[str], str, str]] = [
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
]


def sanitize_readonly_sql(sql: str) -> tuple[str | None, str]:
    """
    校验并规范化只读 SQL。
    返回 (cleaned_sql|None, error_msg)。
    """
    if not sql or not str(sql).strip():
        return None, "缺少 sql 查询语句参数"

    cleaned = _COMMENT_BLOCK.sub(" ", str(sql))
    cleaned = _COMMENT_LINE.sub(" ", cleaned)
    cleaned = cleaned.strip()
    # 仅允许末尾一个分号；中间分号视为多语句
    if ";" in cleaned.rstrip(";"):
        return None, "禁止多语句 SQL（仅允许单条 SELECT/WITH）"
    cleaned = cleaned.rstrip(";").strip()
    if not cleaned:
        return None, "SQL 为空"

    head = cleaned.lstrip().lower()
    if not (head.startswith("select") or head.startswith("with")):
        return None, "仅允许执行 SELECT / WITH…SELECT 只读查询"

    # WITH 必须最终落在 SELECT（粗检：全文含 select）
    if head.startswith("with") and not re.search(r"\bselect\b", cleaned, re.IGNORECASE):
        return None, "WITH 语句必须包含 SELECT"

    if _FORBIDDEN.search(cleaned):
        hit = _FORBIDDEN.search(cleaned)
        word = hit.group(1) if hit else "forbidden"
        return None, f"禁止高危 SQL 关键字：{word}"

    return cleaned, ""


def nl_to_readonly_sql(text: str) -> tuple[str | None, str, str]:
    """
    将常见自然语言映射到白名单只读 SQL。
    返回 (sql|None, label, error_msg)。
    """
    q = str(text or "").strip()
    if not q:
        return None, "", "缺少自然语言查询"
    # 用户直接给了 SELECT
    if re.match(r"^\s*(select|with)\b", q, re.I):
        cleaned, err = sanitize_readonly_sql(q)
        return cleaned, "raw_sql", err
    for pat, sql, label in _NL_SQL_TEMPLATES:
        if pat.search(q):
            return sql, label, ""
    return (
        None,
        "",
        "暂不支持该自然语言查库；可试：有多少订单/商品、按店铺统计、竞品报价、有哪些表；"
        "或传 sql=SELECT ...",
    )


@register_skill
class SafeSqlQueryTool(BaseSkill):
    read_only = True
    side_effect = False
    risk_level = "medium"
    skill_name = "safe_sql_query"
    skill_desc = (
        "安全只读数据库查询：参数 sql，或用 query/user_query 映射白名单模板；"
        "仅支持 SELECT/WITH"
    )

    async def run(self, params: dict) -> SkillResult:
        try:
            sql = params.get("sql")
            label = ""
            if not sql:
                nl = str(
                    params.get("query")
                    or params.get("user_query")
                    or params.get("text")
                    or ""
                )
                sql, label, err = nl_to_readonly_sql(nl)
                if err or not sql:
                    return SkillResult(success=False, error_msg=err or "缺少sql查询语句参数")

            cleaned, err = sanitize_readonly_sql(str(sql))
            if err or cleaned is None:
                return SkillResult(success=False, error_msg=err or "非法 SQL")

            query_args = params.get("params", [])
            query_data = await AsyncPGClient.execute_sql(cleaned, query_args)
            return SkillResult(
                success=True,
                data={
                    "query_result": query_data,
                    "sql": cleaned,
                    "label": label or "sql",
                },
            )
        except Exception as e:
            return SkillResult(success=False, error_msg=f"SQL执行异常：{str(e)}")
