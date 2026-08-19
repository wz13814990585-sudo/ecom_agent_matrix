# test_calc_tool.py
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.core.skill.skill_registry import exec_skill
# 必须 import 工具模块，@register_skill 装饰器才会把 profit_calc 注册进容器
from ecom_agent_matrix.modules.skills import calc_tool  # noqa: F401

async def test_calc():
    res = await exec_skill("profit_calc", {
        "cost": 22.5,
        "shipping": 9.2,
        "commission_rate": 0.078,
        "sell_price": 64.99
    })
    print(res.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(test_calc())