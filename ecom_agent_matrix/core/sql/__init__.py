"""共享的 SQL 安全策略。"""

from .readonly_policy import nl_to_readonly_sql, sanitize_readonly_sql

__all__ = ["nl_to_readonly_sql", "sanitize_readonly_sql"]
