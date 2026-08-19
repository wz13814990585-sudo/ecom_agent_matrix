"""Skill 注册冒烟：列出全部已注册 skill，并跑少量无外部依赖用例。"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.core.skill.skill_registry import exec_skill, skill_container
import ecom_agent_matrix.modules.skills  # noqa: F401


async def batch_test():
    names = sorted(skill_container.keys())
    print(f"已注册 skills ({len(names)}): {', '.join(names)}")
    assert "competitor_price" in names
    assert "safe_sql_query" in names
    assert "ops_report" in names
    assert "goods_catalog" in names

    profit = await exec_skill(
        "profit_calc",
        {"cost": 19.9, "shipping": 8.5, "commission_rate": 0.08, "sell_price": 59.99},
    )
    assert profit.success, profit.error_msg

    risk = await exec_skill(
        "order_risk_check",
        {"order_no": "ORD20260814001", "total_amount": 680, "buy_count": 30},
    )
    assert risk.success, risk.error_msg

    bad = await exec_skill("safe_sql_query", {"sql": "DELETE FROM ecom_goods"})
    assert bad.success is False
    assert "仅允许" in (bad.error_msg or "") or "禁止" in (bad.error_msg or "")

    multi = await exec_skill(
        "safe_sql_query",
        {"sql": "SELECT 1; DROP TABLE ecom_goods"},
    )
    assert multi.success is False


if __name__ == "__main__":
    asyncio.run(batch_test())
    print("✅ skill 冒烟通过")
