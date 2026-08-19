# test_skill_framework.py
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.core.skill.skill_registry import exec_skill

# 导入 Skill 模块以触发 @register_skill 注册
from ecom_agent_matrix.modules.skills import calc_tool  # noqa: F401
from ecom_agent_matrix.modules.skills import translate_tool  # noqa: F401


async def test_all_skill():
    # 1. 利润计算（calc_tool.profit_calc 在 calc_tool 里注册为 profit_calc）
    res1 = await exec_skill("profit_calc", {
        "cost": 22.5,
        "shipping": 9.2,
        "commission_rate": 0.078,
        "sell_price": 64.99,
    })
    print("利润计算结果：", res1.model_dump())

    # 2. LLM 翻译（translate_tool.py）
    translate_cases = [
        {"text": "我是大帅哥", "target_lang": "en"},
        {"text": "平价海边连衣裙", "target_lang": "en"},
        {"text": "防水户外背包", "target_lang": "es"},
        {"text": "free returns", "target_lang": "zh"},
        {"text": "affordable beach dress", "target_lang": "fr"},
    ]
    for case in translate_cases:
        res = await exec_skill("text_translate", case)
        print("翻译结果：", res.model_dump())

    # 3. 不存在的工具
    res3 = await exec_skill("stock_predict", {})
    print("错误工具返回：", res3.model_dump())


if __name__ == "__main__":
    asyncio.run(test_all_skill())
