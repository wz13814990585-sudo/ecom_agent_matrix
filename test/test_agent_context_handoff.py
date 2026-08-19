"""Phase 2C TaskContext Agent handoff closure tests。"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.core.tasking import TaskContext, normalize_task_context
from ecom_agent_matrix.modules.agent_cluster.exec_agent import run_exec
from ecom_agent_matrix.modules.agent_cluster.query_agent import _ensure_sku, run_query


def test_query_handler_receives_task_context():
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.query_agent.handle_goods", new=AsyncMock(return_value=(True, "", {}))) as handler:
            await run_query(normalize_task_context({"task_type": "goods_search", "query": "bag"}))
        return handler.await_args.args[0]
    assert isinstance(asyncio.run(scenario()), TaskContext)


def test_exec_handler_receives_task_context():
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.exec_agent.handle_ad", new=AsyncMock(return_value=(True, "", {}))) as handler:
            await run_exec(normalize_task_context({"task_type": "ad_optimize", "spend": 10}))
        return handler.await_args.args[0]
    assert isinstance(asyncio.run(scenario()), TaskContext)


def test_ensure_sku_returns_new_context_without_mutating_original():
    original = normalize_task_context({"query": "防水背包库存", "nested": {"x": [1]}})
    before = deepcopy(original.model_dump())
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.query_agent.handle_goods", new=AsyncMock(return_value=(
            True, "", {"best_sku": "SKU-1", "candidates": [{"sku": "SKU-1"}]},
        ))):
            return await _ensure_sku(original)
    enriched, early = asyncio.run(scenario())
    assert early is None
    assert enriched is not original
    assert enriched.sku == "SKU-1"
    assert original.model_dump() == before


def test_run_query_dict_remains_compatible_after_handoff():
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.query_agent.handle_goods", new=AsyncMock(return_value=(True, "", {}))):
            return await run_query({"task_type": "goods_search", "query": "bag"})
    assert asyncio.run(scenario())[0] is True


def test_run_exec_dict_remains_compatible_after_handoff():
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.exec_agent.handle_report", new=AsyncMock(return_value=(True, "", {}))):
            return await run_exec({"task_type": "ops_report", "report_type": "sales"})
    assert asyncio.run(scenario())[0] is True
