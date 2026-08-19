"""商品目录查询 skill / 路由单测。"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ecom_agent_matrix.modules.skills  # noqa: F401
from ecom_agent_matrix.config.constants import AGENT_QUERY
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.modules.agent_cluster.master_planner import (
    infer_task_type_from_query,
    plan_sub_tasks_rules,
    react_decide_rules,
)
from ecom_agent_matrix.modules.skills.goods_catalog import is_catalog_query, wants_full_catalog


def test_is_catalog_query():
    assert is_catalog_query("我现在想要查询我的数据库里面有哪些商品")
    assert is_catalog_query("有多少商品")
    assert is_catalog_query("列出全部商品")
    assert is_catalog_query("how many products in the catalog")
    assert not is_catalog_query("搜索防水户外背包")
    assert not is_catalog_query("为防水背包写tiktok文案")


def test_wants_full_catalog():
    assert wants_full_catalog("展示全部商品")
    assert wants_full_catalog("列出所有商品")
    assert wants_full_catalog("查询数据库有哪些商品")
    assert not wants_full_catalog("有多少商品")


def test_infer_catalog_intent():
    assert infer_task_type_from_query("查询数据库里面有哪些商品") == "goods_catalog"
    assert infer_task_type_from_query("商品库一共有多少商品") == "goods_catalog"
    plan = plan_sub_tasks_rules(
        {"query": "列出商品", "task_type": "goods_catalog"},
        [],
    )
    assert plan.sub_tasks[0]["target_agent"] == AGENT_QUERY
    assert plan.sub_tasks[0]["payload"].get("mode") == "catalog"


def test_react_catalog_finishes():
    working = {"query": "有多少商品", "task_type": "goods_catalog", "mode": "catalog"}
    obs = {
        "agent": AGENT_QUERY,
        "success": True,
        "data": {
            "mode": "catalog",
            "total": 100,
            "summary": "商品库共 100 件",
            "items": [{"sku": "SKU-1"}],
        },
        "error_msg": "",
    }
    d = react_decide_rules(working, [obs], [AGENT_QUERY])
    assert d.action == "finish"
    assert "100" in (d.final_answer or "")


async def test_goods_catalog_skill_mocked():
    async def fake_sql(sql, params=None):
        if "COUNT" in sql.upper():
            return [(100,)]
        return [
            ("SKU-A", "背包A", "Bag A", "bag", 39.9, 12, "demo_store", "我的模拟独立站", True),
            ("SKU-B", "帽子B", "Hat B", "hat", 19.9, 5, "demo_store", "我的模拟独立站", True),
        ]

    with patch(
        "ecom_agent_matrix.modules.skills.goods_catalog.AsyncPGClient.execute_sql",
        new=AsyncMock(side_effect=fake_sql),
    ):
        res = await exec_skill("goods_catalog", {"limit": 20, "offset": 0})
    assert res.success is True
    assert res.data["total"] == 100
    assert res.data["count"] == 2
    assert "100" in res.data["summary"]


async def test_goods_catalog_list_all_uses_total():
    rows = [
        (f"SKU-{i}", f"商品{i}", f"P{i}", "bag", 10.0, i, "demo_store", "我的模拟独立站", True)
        for i in range(100)
    ]

    async def fake_sql(sql, params=None):
        if "COUNT" in sql.upper():
            return [(100,)]
        # list_all 后 limit 应为 100
        assert params is not None and params[-2] == 100
        return rows

    with patch(
        "ecom_agent_matrix.modules.skills.goods_catalog.AsyncPGClient.execute_sql",
        new=AsyncMock(side_effect=fake_sql),
    ):
        res = await exec_skill(
            "goods_catalog",
            {"query": "展示全部商品", "list_all": True},
        )
    assert res.success is True
    assert res.data["count"] == 100
    assert res.data["list_all"] is True
    assert res.data["truncated"] is False


if __name__ == "__main__":
    test_is_catalog_query()
    test_wants_full_catalog()
    test_infer_catalog_intent()
    test_react_catalog_finishes()
    asyncio.run(test_goods_catalog_skill_mocked())
    asyncio.run(test_goods_catalog_list_all_uses_total())
    print("✅ goods_catalog tests ok")
