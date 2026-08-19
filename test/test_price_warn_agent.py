"""price_warn handler：缺价询价 + 监控 + LLM 解读（全 mock）。"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.core.skill.base_skill import SkillResult
import ecom_agent_matrix.modules.agent_cluster.handlers.competitor as price_mod


async def test_price_warn_fetches_price_and_advice():
    async def fake_exec(name: str, params: dict):
        if name == "competitor_price":
            return SkillResult(
                success=True,
                data={
                    "compete_price": 41.5,
                    "price_source": "demo_synthesize",
                    "source_ref": "demo://x",
                },
            )
        if name == "price_monitor":
            return SkillResult(
                success=True,
                data={
                    "is_trigger_warn": True,
                    "warn_message": "大幅降价",
                    "current_price_offset": -12.0,
                    "warn_threshold": -10,
                },
            )
        return SkillResult(success=False, error_msg=f"unexpected skill {name}")

    mem = MagicMock()
    mem.recall = AsyncMock(return_value=[])
    mem.safe_save_memory = AsyncMock()

    payload = {
        "target_sku": "SKU-BAG-001",
        "competitor": "Temu",
        "query": "监控 Temu 上 SKU-BAG-001",
    }

    with patch(
        "ecom_agent_matrix.modules.agent_cluster.handlers.competitor.exec_skill",
        side_effect=fake_exec,
    ), patch(
        "ecom_agent_matrix.modules.agent_cluster.handlers.competitor._mem",
        return_value=mem,
    ), patch(
        "ecom_agent_matrix.modules.agent_cluster.handlers.competitor.llm_explain",
        new=AsyncMock(return_value=("建议观望并核对利润。", "deepseek", "")),
    ):
        ok, err, data = await price_mod.handle_price_warn(payload)

    assert ok is True
    assert not err
    assert data["compete_price"] == 41.5
    assert data["is_trigger_warn"] is True
    assert data["advice"] == "建议观望并核对利润。"
    print("✅ price_warn handler 询价+监控+解读测试通过")


if __name__ == "__main__":
    asyncio.run(test_price_warn_fetches_price_and_advice())
