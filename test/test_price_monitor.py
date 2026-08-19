"""price_monitor Decimal 兼容。"""
import asyncio
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ecom_agent_matrix.modules.skills  # noqa: F401
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.modules.skills.price_monitor import _as_float


def test_as_float_decimal():
    assert _as_float(Decimal("12.50")) == 12.5
    assert _as_float(9.9) == 9.9


async def test_price_monitor_handles_decimal_min():
    async def fake_sql(sql, params=None):
        if "INSERT" in sql.upper():
            return [(101,)]
        if "MIN" in sql.upper():
            return [(Decimal("40.00"),)]
        return []

    with patch(
        "ecom_agent_matrix.modules.skills.price_monitor.AsyncPGClient.execute_sql",
        new=AsyncMock(side_effect=fake_sql),
    ):
        res = await exec_skill(
            "price_monitor",
            {"target_sku": "SKU-BAG-001", "competitor": "Temu", "compete_price": 43.99},
        )
    assert res.success is True
    assert res.data["history_min_compete_price"] == 40.0
    assert res.data["current_price_offset"] == round(43.99 - 40.0, 2)


if __name__ == "__main__":
    test_as_float_decimal()
    asyncio.run(test_price_monitor_handles_decimal_min())
    print("✅ price_monitor decimal ok")
