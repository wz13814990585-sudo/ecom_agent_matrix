"""一键初始化所有数据库表。"""
import asyncio
import sys
from pathlib import Path

# 允许直接运行脚本：python ecom_agent_matrix/scripts/init_db.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.db.base import AsyncPGClient


def split_sql_statements(sql_text: str) -> list[str]:
    """Split SQL without breaking quoted strings or PostgreSQL dollar blocks."""
    statements: list[str] = []
    buffer: list[str] = []
    index = 0
    quote = ""
    dollar_tag = ""
    line_comment = False
    block_comment = False
    length = len(sql_text)
    while index < length:
        char = sql_text[index]
        pair = sql_text[index:index + 2]
        if line_comment:
            if char == "\n":
                line_comment = False
                buffer.append(char)
            index += 1
            continue
        if block_comment:
            if pair == "*/":
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if dollar_tag:
            if sql_text.startswith(dollar_tag, index):
                buffer.append(dollar_tag)
                index += len(dollar_tag)
                dollar_tag = ""
            else:
                buffer.append(char)
                index += 1
            continue
        if quote:
            buffer.append(char)
            if char == quote:
                if index + 1 < length and sql_text[index + 1] == quote:
                    buffer.append(sql_text[index + 1])
                    index += 2
                    continue
                quote = ""
            index += 1
            continue
        if pair == "--":
            line_comment = True
            index += 2
            continue
        if pair == "/*":
            block_comment = True
            index += 2
            continue
        if char in {"'", '"'}:
            quote = char
            buffer.append(char)
            index += 1
            continue
        if char == "$":
            end = sql_text.find("$", index + 1)
            if end != -1:
                candidate = sql_text[index:end + 1]
                tag_body = candidate[1:-1]
                if not tag_body or tag_body.replace("_", "a").isalnum():
                    dollar_tag = candidate
                    buffer.append(candidate)
                    index = end + 1
                    continue
        if char == ";":
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
        else:
            buffer.append(char)
        index += 1
    statement = "".join(buffer).strip()
    if statement:
        statements.append(statement)
    return statements


def ensure_database():
    """若业务库不存在则自动创建。"""
    conn = psycopg2.connect(
        host=settings.PG_HOST,
        port=settings.PG_PORT,
        user=settings.PG_USER,
        password=settings.PG_PWD,
        dbname="postgres",
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s;",
                (settings.PG_DB,),
            )
            exists = cur.fetchone()
            if not exists:
                cur.execute(f'CREATE DATABASE "{settings.PG_DB}";')
                print(f"✅ 已创建数据库: {settings.PG_DB}")
            else:
                print(f"ℹ️  数据库已存在: {settings.PG_DB}")
    finally:
        conn.close()


async def execute_sql_file(sql_path: Path):
    """逐条执行 SQL 文件中的语句。"""
    sql_text = sql_path.read_text(encoding="utf-8")
    statements = split_sql_statements(sql_text)
    for stmt in statements:
        try:
            await AsyncPGClient.execute_sql(stmt)
        except Exception as exc:
            err = str(exc).lower()
            # 索引/扩展已存在时跳过，支持重复初始化
            if "already exists" in err:
                print(f"⚠️  跳过已存在对象: {stmt.splitlines()[0][:60]}...")
                continue
            raise
    print(f"✅ 已执行: {sql_path.name} ({len(statements)} 条语句)")


async def init_all_table():
    sql_dir = Path(__file__).parent.parent / "db"
    ensure_database()
    await execute_sql_file(sql_dir / "business_tables.sql")
    await execute_sql_file(sql_dir / "vector_tables.sql")
    await AsyncPGClient.close()
    print("✅ 全部业务表、向量表初始化完成！")


if __name__ == "__main__":
    asyncio.run(init_all_table())
