"""Phase 2C-1 Goods parser / workflow 测试。"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.core.skill.base_skill import SkillResult
from ecom_agent_matrix.core.tasking import TaskContext, WorkflowResult, normalize_task_context
from ecom_agent_matrix.core.tasking.result import SKILL_FAILED
from ecom_agent_matrix.modules.agent_cluster.handlers.goods import (
    handle_goods,
    run_goods_workflow,
)
from ecom_agent_matrix.modules.parsers.goods import parse_goods_request


def test_goods_explicit_product_name_has_priority():
    ctx = normalize_task_context(
        {"query": "查询鞋子的库存", "product_name": " 防水背包 "}
    )
    assert parse_goods_request(ctx).product_name == "防水背包"


def test_goods_product_name_is_extracted_and_cleaned_from_query():
    ctx = normalize_task_context({"query": "帮我看看防水背包的库存"})
    assert parse_goods_request(ctx).product_name == "防水背包"


def test_goods_catalog_mode_is_deterministic():
    assert parse_goods_request(normalize_task_context({"query": "列出全部商品"})).mode == "catalog"
    assert (
        parse_goods_request(normalize_task_context({"task_type": "goods_catalog"})).mode
        == "catalog"
    )


def test_goods_search_mode_is_deterministic():
    request = parse_goods_request(normalize_task_context({"query": "防水背包"}))
    assert request.mode == "search"
    assert request.product_name == "防水背包"


def test_goods_list_all_and_top_k_are_parsed():
    catalog = parse_goods_request(
        normalize_task_context({"query": "列出全部商品", "top_k": 12})
    )
    search = parse_goods_request(
        normalize_task_context({"query": "防水背包", "top_k": 7})
    )
    assert catalog.list_all is True
    assert catalog.limit is None
    assert search.top_k == 7


def test_goods_parser_does_not_reread_conflicting_user_query_alias():
    ctx = normalize_task_context(
        {"query": "防水背包", "user_query": "列出全部商品"}
    )
    request = parse_goods_request(ctx)
    assert request.mode == "search"
    assert request.product_name == "防水背包"


def test_goods_parser_does_not_modify_context_or_params():
    ctx = normalize_task_context(
        {"query": " 防水背包 ", "filters": {"categories": ["bags"]}}
    )
    before = ctx.model_dump()
    params_before = deepcopy(ctx.params)
    parse_goods_request(ctx)
    assert ctx.model_dump() == before
    assert ctx.params == params_before


def test_goods_catalog_workflow_builds_exact_skill_params():
    skill_result = SkillResult(
        success=True,
        data={"total": 2, "count": 2, "items": [{"sku": "A"}, {"sku": "B"}]},
    )

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.goods.exec_skill",
            new=AsyncMock(return_value=skill_result),
        ) as execute:
            result = await run_goods_workflow(
                {
                    "query": "商品目录",
                    "mode": "catalog",
                    "offset": 3,
                    "limit": 10,
                    "category": "bags",
                    "order_by": "stock",
                    "store_id": "S-1",
                }
            )
        return result, execute.await_args.args

    result, args = asyncio.run(scenario())
    assert result.success is True
    assert args == (
        "goods_catalog",
        {
            "offset": 3,
            "category": "bags",
            "order_by": "stock",
            "query": "商品目录",
            "list_all": True,
            "store_id": "S-1",
            "limit": 10,
        },
    )
    assert result.metadata["workflow"] == "goods"


def test_goods_search_workflow_builds_exact_skill_params():
    skill_result = SkillResult(
        success=True,
        data={
            "candidates": [{"sku": "SKU-1"}],
            "best_sku": "SKU-1",
            "count": 1,
            "match_mode": "literal",
        },
    )

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.goods.exec_skill",
            new=AsyncMock(return_value=skill_result),
        ) as execute:
            result = await run_goods_workflow(
                normalize_task_context({"query": "防水背包", "top_k": 8})
            )
        return result, execute.await_args.args

    result, args = asyncio.run(scenario())
    assert result.success is True
    assert args == ("goods_sku_search", {"product_name": "防水背包", "top_k": 8})
    assert result.data["best_sku"] == "SKU-1"


def test_goods_workflow_preserves_skill_error_code():
    failure = SkillResult(
        success=False,
        error_code="PERMISSION_DENIED",
        error_msg="denied",
    )

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.goods.exec_skill",
            new=AsyncMock(return_value=failure),
        ):
            return await run_goods_workflow({"query": "防水背包"})

    result = asyncio.run(scenario())
    assert result.success is False
    assert result.error_code == SKILL_FAILED
    assert result.metadata["skill_error_code"] == "PERMISSION_DENIED"


def test_goods_legacy_handler_accepts_dict_and_returns_tuple():
    success = SkillResult(
        success=True,
        data={"candidates": [{"sku": "SKU-1"}], "best_sku": "SKU-1", "count": 1},
    )

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.goods.exec_skill",
            new=AsyncMock(return_value=success),
        ):
            return await handle_goods({"query": "防水背包"})

    legacy = asyncio.run(scenario())
    assert isinstance(legacy, tuple)
    assert legacy[0] is True


def test_workflow_result_defaults_and_legacy_tuple_are_safe():
    first = WorkflowResult(success=True, data={"nested": {"value": 1}})
    second = WorkflowResult(success=True)
    first.metadata["workflow"] = "goods"
    legacy = first.as_legacy_tuple()
    legacy[2]["nested"]["value"] = 2

    assert second.data == {}
    assert second.metadata == {}
    assert first.data["nested"]["value"] == 1
    assert legacy[:2] == (True, "")
