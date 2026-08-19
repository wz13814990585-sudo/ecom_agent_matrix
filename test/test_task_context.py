"""Phase 2B：TaskContext / TaskNormalizer 回归测试。"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ecom_agent_matrix.config.constants import AGENT_EXEC, AGENT_QUERY
from ecom_agent_matrix.core.tasking import (
    TaskContext,
    ensure_task_context,
    normalize_task_context,
)
from ecom_agent_matrix.modules.agent_cluster.exec_agent import infer_exec_kind, run_exec
from ecom_agent_matrix.modules.agent_cluster.query_agent import infer_query_kind, run_query


@pytest.mark.parametrize("field", ["query", "user_query", "text", "message"])
def test_query_aliases_are_normalized(field: str):
    ctx = normalize_task_context({field: "  查询库存  "})
    assert ctx.query == "查询库存"


def test_string_content_is_query_alias_but_mapping_content_is_not():
    assert normalize_task_context({"content": "  查询订单  "}).query == "查询订单"
    assert normalize_task_context({"content": {"query": "不可递归读取"}}).query == ""


@pytest.mark.parametrize("field", ["sku", "target_sku", "best_sku"])
def test_sku_aliases_are_normalized(field: str):
    ctx = normalize_task_context({field: "  SKU-1  "})
    assert ctx.sku == "SKU-1"


def test_product_name_and_language_aliases_are_normalized():
    ctx = normalize_task_context({"goods_name": "  防水背包  ", "language": " zh-CN "})
    assert ctx.product_name == "防水背包"
    assert ctx.lang == "zh-CN"


def test_canonical_fields_win_over_aliases():
    ctx = normalize_task_context(
        {
            "query": "正式请求",
            "user_query": "旧请求",
            "sku": "SKU-A",
            "target_sku": "SKU-B",
            "product_name": "正式商品",
            "goods_name": "旧商品",
            "lang": "zh",
            "language": "en",
        }
    )
    assert ctx.query == "正式请求"
    assert ctx.sku == "SKU-A"
    assert ctx.product_name == "正式商品"
    assert ctx.lang == "zh"


def test_whitespace_is_stripped_and_empty_strings_become_none():
    ctx = normalize_task_context(
        {
            "query": "   ",
            "user_query": "  fallback  ",
            "sku": " ",
            "product_name": "",
            "query_kind": " stock ",
        }
    )
    assert ctx.query == "fallback"
    assert ctx.sku is None
    assert ctx.product_name is None
    assert ctx.query_kind == "stock"
    assert normalize_task_context({"query": " "}).query == ""
    assert "sku" not in ctx.to_payload()
    assert "product_name" not in ctx.to_payload()


def test_params_preserve_unknown_business_fields():
    ctx = normalize_task_context(
        {
            "query": "优化广告",
            "campaign_id": "C-1",
            "spend": 120,
            "roas": 1.8,
            "custom_business_field": "abc",
        }
    )
    assert ctx.query == "优化广告"
    assert ctx.campaign_id == "C-1"
    assert ctx.params["spend"] == 120
    assert ctx.params["roas"] == 1.8
    assert ctx.params["custom_business_field"] == "abc"


def test_normalizer_deep_copies_input_payload():
    raw = {"query": "库存", "filters": {"stores": ["S-1"]}}
    ctx = normalize_task_context(raw)
    ctx.params["filters"]["stores"].append("S-2")
    assert raw["filters"]["stores"] == ["S-1"]

    raw["filters"]["stores"].append("S-3")
    assert ctx.params["filters"]["stores"] == ["S-1", "S-2"]


def test_to_payload_preserves_business_fields_and_returns_deep_copy():
    ctx = normalize_task_context(
        {"user_query": "优化广告", "target_sku": "SKU-1", "options": {"dry_run": True}}
    )
    first = ctx.to_payload()
    first["options"]["dry_run"] = False
    second = ctx.to_payload()

    assert second["query"] == "优化广告"
    assert second["sku"] == "SKU-1"
    assert second["options"] == {"dry_run": True}


def test_envelope_ids_only_come_from_explicit_arguments():
    ctx = normalize_task_context(
        {"task_id": "fake-task", "correlation_id": "fake-corr", "source_agent": "fake"},
        task_id=" root-task ",
        correlation_id=" corr-real ",
        source_agent=AGENT_QUERY,
    )
    assert ctx.task_id == "root-task"
    assert ctx.correlation_id == "corr-real"
    assert ctx.source_agent == AGENT_QUERY


def test_fake_envelope_ids_do_not_leak_to_business_payload():
    ctx = normalize_task_context(
        {"query": "test", "task_id": "fake-task", "correlation_id": "fake-corr"},
        task_id="real-task",
        correlation_id="real-corr",
    )
    payload = ctx.to_payload()
    assert "task_id" not in payload
    assert "correlation_id" not in payload
    assert "source_agent" not in payload


def test_task_type_alias_and_explicit_kinds_are_preserved():
    ctx = normalize_task_context(
        {
            "_inferred_task_type": " customer_service ",
            "query_kind": " data_check ",
            "exec_kind": " crm ",
        }
    )
    assert ctx.task_type == "customer_service"
    assert ctx.query_kind == "data_check"
    assert ctx.exec_kind == "crm"


def test_with_updates_returns_independent_context():
    original = normalize_task_context({"target_sku": "SKU-1", "nested": {"value": 1}})
    updated = original.with_updates(sku="SKU-2")
    updated.params["nested"]["value"] = 2

    assert original.sku == "SKU-1"
    assert updated.sku == "SKU-2"
    assert original.params["nested"]["value"] == 1


def test_ensure_task_context_keeps_context_and_normalizes_dict():
    ctx = TaskContext(query="existing")
    assert ensure_task_context(ctx) is ctx
    assert ensure_task_context({"text": "legacy"}).query == "legacy"


def test_run_query_dict_remains_compatible():
    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.query_agent.handle_goods",
            new=AsyncMock(return_value=(True, "", {"query_kind": "goods"})),
        ) as handler:
            result = await run_query({"task_type": "goods_search", "user_query": " 背包 "})
        assert handler.await_args.args[0]["query"] == "背包"
        return result

    assert asyncio.run(scenario())[0] is True


def test_run_query_task_context_works():
    async def scenario():
        ctx = normalize_task_context({"task_type": "goods_search", "text": " 鞋子 "})
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.query_agent.handle_goods",
            new=AsyncMock(return_value=(True, "", {"query_kind": "goods"})),
        ) as handler:
            result = await run_query(ctx)
        assert handler.await_args.args[0]["query"] == "鞋子"
        return result

    assert asyncio.run(scenario())[0] is True


def test_run_exec_dict_remains_compatible():
    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.exec_agent.handle_ad",
            new=AsyncMock(return_value=(True, "", {"exec_kind": "ad"})),
        ) as handler:
            result = await run_exec({"task_type": "ad_optimize", "text": " 优化广告 "})
        assert handler.await_args.args[0]["query"] == "优化广告"
        return result

    assert asyncio.run(scenario())[0] is True


def test_run_exec_task_context_works_and_preserves_root_task_id_for_crm():
    async def scenario():
        ctx = normalize_task_context(
            {"task_type": "customer_service", "message": " 回复客户 "},
            task_id="root-1",
        )
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.exec_agent.handle_crm",
            new=AsyncMock(return_value=(True, "", {"exec_kind": "crm"})),
        ) as handler:
            result = await run_exec(ctx)
        assert handler.await_args.args[0]["query"] == "回复客户"
        assert handler.await_args.kwargs["task_id"] == "root-1"
        return result

    assert asyncio.run(scenario())[0] is True


def test_query_routes_are_consistent_for_legacy_aliases_and_context():
    for payload in (
        {"query": "查询库存"},
        {"user_query": "查询库存"},
        {"text": "查询库存"},
        {"message": "查询库存"},
    ):
        assert infer_query_kind(payload) == "stock"
        assert infer_query_kind(normalize_task_context(payload)) == "stock"
    assert infer_query_kind({"_inferred_task_type": "order_query"}) == "data_check"


def test_exec_routes_are_consistent_for_legacy_aliases_and_context():
    for payload in (
        {"query": "帮我回复退款客户"},
        {"user_query": "帮我回复退款客户"},
        {"text": "帮我回复退款客户"},
        {"message": "帮我回复退款客户"},
    ):
        assert infer_exec_kind(payload) == "crm"
        assert infer_exec_kind(normalize_task_context(payload)) == "crm"
    assert infer_exec_kind({"_inferred_task_type": "ad_optimize"}) == "ad"
