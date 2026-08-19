"""数据库异步连接基类。"""
# db/base.py
import aiopg
from ecom_agent_matrix.config.settings import settings

class AsyncPGClient:
    # 类变量：全局连接池单例，整个项目只创建1次连接池，复用连接，减少IO开销
    _pool = None

    @classmethod
    async def get_pool(cls):
        """懒加载：第一次使用时创建连接池，后续直接复用"""
        if cls._pool is None:
            # 拼接数据库连接字符串dsn
            dsn = (
                f"host={settings.PG_HOST} port={settings.PG_PORT} "
                f"user={settings.PG_USER} password={settings.PG_PWD} dbname={settings.PG_DB}"
            )
            # 创建异步连接池，自动管理连接创建/释放
            cls._pool = await aiopg.create_pool(dsn)
        return cls._pool

    @classmethod
    async def execute_sql(cls, sql: str, params: list = None):
        """通用执行SQL方法，支持查询/插入/更新"""
        pool = await cls.get_pool()
        # 从池中取出一条数据库连接
        async with pool.acquire() as conn:
            # 创建游标执行SQL
            async with conn.cursor() as cur:
                # params防止SQL注入，参数化查询核心安全方案
                await cur.execute(sql, params or [])
                if cur.description:
                    return await cur.fetchall()
                return []

    @classmethod
    async def close(cls):
        """项目关闭时释放连接池，优雅退出"""
        if cls._pool:
            cls._pool.close()
            await cls._pool.wait_closed()