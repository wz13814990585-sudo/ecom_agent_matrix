"""Query / Exec 子 Agent 内部意图与 SKU 补全。"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.modules.agent_cluster.exec_agent import infer_exec_kind
from ecom_agent_matrix.modules.agent_cluster.query_agent import infer_query_kind, run_query


def test_infer_query_kind():
    assert infer_query_kind({"query": "列出全部商品"}) == "goods"
    assert infer_query_kind({"query": "防水背包备货多少"}) == "stock"
    assert infer_query_kind({"query": "Temu 竞品比价"}) == "competitor"
    assert infer_query_kind({"query": "检查商品主数据"}) == "data_check"
    assert infer_query_kind({"task_type": "ad_query", "query": "查广告消耗"}) == "data_check"


def test_infer_exec_kind():
    assert infer_exec_kind({"query": "调整 Meta 广告出价", "spend": 100}) == "ad"
    assert infer_exec_kind({"query": "生成近 7 天运营报表"}) == "report"
    assert infer_exec_kind({"query": "触发订单风控", "order_no": "ORD-1"}) == "risk"
    assert infer_exec_kind({"query": "为「背包」写 tiktok 文案"}) == "social"


async def test_query_stock_resolves_sku_internally():
    async def fake_goods(payload):
        return True, "", {"best_sku": "SKU-BAG-001", "candidates": [{"sku": "SKU-BAG-001"}]}

    async def fake_stock(payload):
        assert payload.get("sku") == "SKU-BAG-001"
        return True, "", {"query_kind": "stock", "sku": payload["sku"], "stock_predict_result": {"suggest_stock_amount": 12}}

    with patch(
        "ecom_agent_matrix.modules.agent_cluster.query_agent.handle_goods",
        new=AsyncMock(side_effect=fake_goods),
    ), patch(
        "ecom_agent_matrix.modules.agent_cluster.query_agent.handle_stock",
        new=AsyncMock(side_effect=fake_stock),
    ):
        ok, err, data = await run_query({"query": "防水户外背包需要备货多少", "task_type": "stock_analysis"})
    assert ok is True
    assert data["sku"] == "SKU-BAG-001"
    assert data["stock_predict_result"]["suggest_stock_amount"] == 12


if __name__ == "__main__":
    test_infer_query_kind()
    test_infer_exec_kind()
    asyncio.run(test_query_stock_resolves_sku_internally())
    print("✅ query/exec 意图测试通过")
