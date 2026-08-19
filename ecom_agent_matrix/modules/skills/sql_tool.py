"""安全只读 SQL 查询工具（由 data_check Agent 按需调用：payload.sql / custom_sql / NL）。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ecom_agent_matrix.core.sql import nl_to_readonly_sql, sanitize_readonly_sql
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill
from ecom_agent_matrix.db.base import AsyncPGClient
from ecom_agent_matrix.config.settings import settings

class SafeSqlQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sql: str | None = None
    query: str | None = None
    user_query: str | None = None
    text: str | None = None
    params: list[Any] | dict[str, Any] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_query(self) -> "SafeSqlQueryInput":
        if not any((self.sql, self.query, self.user_query, self.text)):
            raise ValueError("必须提供 sql、query、user_query 或 text")
        return self


class SafeSqlQueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_result: list[Any]
    sql: str
    label: str
    truncated: bool = False


@register_skill
class SafeSqlQueryTool(BaseSkill):
    read_only = True
    side_effect = False
    risk_level = "medium"
    timeout_seconds = 15.0
    idempotent = True
    input_model = SafeSqlQueryInput
    output_model = SafeSqlQueryOutput
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
            query_data, truncated = await AsyncPGClient.execute_read_bounded(
                cleaned,
                query_args,
                max_rows=settings.DB_READ_MAX_ROWS,
            )
            return SkillResult(
                success=True,
                data={
                    "query_result": query_data,
                    "sql": cleaned,
                    "label": label or "sql",
                    "truncated": truncated,
                },
            )
        except Exception as exc:
            return SkillResult(success=False, error_msg=f"SQL执行异常：{type(exc).__name__}")
