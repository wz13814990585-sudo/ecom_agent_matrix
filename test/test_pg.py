import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.db.base import AsyncPGClient

async def test_conn():
    res = await AsyncPGClient.execute_sql("SELECT 1;")
    print("数据库连接成功", res)
    await AsyncPGClient.close()

if __name__ == "__main__":
    asyncio.run(test_conn())