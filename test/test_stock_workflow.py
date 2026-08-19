"""Phase 2C-2A Stock parser / workflow 测试。"""
from __future__ import annotations

import asyncio
import inspect
from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from ecom_agent_matrix.core.skill.base_skill import SkillResult
from ecom_agent_matrix.core.tasking import normalize_task_context
from ecom_agent_matrix.core.tasking.result import MISSING_SKU
from ecom_agent_matrix.modules.agent_cluster.handlers import stock as stock_handler
from ecom_agent_matrix.modules.agent_cluster.handlers.stock import handle_stock, run_stock_workflow
from ecom_agent_matrix.modules.parsers.stock import parse_stock_request
from ecom_agent_matrix.modules.skills.stock_predict import StockPredictTool


def test_stock_canonical_sku_wins_over_query():
    ctx = normalize_task_context({"sku": "SKU-CANON", "query": "查询 SKU-OTHER 库存"})
    assert parse_stock_request(ctx).sku == "SKU-CANON"


@pytest.mark.parametrize("days", [0, 91])
def test_stock_predict_days_validation(days: int):
    with pytest.raises(ValidationError):
        parse_stock_request(normalize_task_context({"sku": "SKU-1", "predict_days": days}))


def test_stock_parser_does_not_modify_context():
    ctx = normalize_task_context({"query": "SKU-ABC 库存", "nested": {"x": [1]}})
    before = deepcopy(ctx.model_dump())
    parse_stock_request(ctx)
    assert ctx.model_dump() == before


def test_stock_missing_sku_has_structured_error():
    result = asyncio.run(run_stock_workflow({"query": "看看库存"}))
    assert result.success is False
    assert result.error_code == MISSING_SKU


def test_stock_predict_receives_only_current_fact_parameters():
    success = SkillResult(
        success=True,
        data={"daily_avg_sales": 2, "suggest_stock_amount": 17},
    )

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.stock.exec_skill",
            new=AsyncMock(return_value=success),
        ) as execute, patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.stock.llm_explain",
            new=AsyncMock(return_value=("advice", "template", "")),
        ):
            result = await run_stock_workflow({"sku": "SKU-1", "predict_days": 14})
        return result, execute.await_args.args

    result, args = asyncio.run(scenario())
    assert result.success is True
    assert args == ("stock_predict", {"sku": "SKU-1", "predict_days": 14})


def test_stock_workflow_no_long_memory_read_or_write():
    source = inspect.getsource(stock_handler)
    assert "AgentLongVectorMemory" not in source
    assert ".recall(" not in source
    assert "safe_save_memory" not in source


def test_historical_suggestion_does_not_change_stock_prediction():
    async def run(history):
        with patch(
            "ecom_agent_matrix.modules.skills.stock_predict.AsyncPGClient.execute_sql",
            new=AsyncMock(return_value=[[300]]),
        ):
            return await StockPredictTool().run(
                {"sku": "SKU-1", "predict_days": 7, "history_records": history}
            )

    without_history = asyncio.run(run([]))
    with_history = asyncio.run(
        run([{"meta": {"suggest_stock_amount": 999999}}])
    )
    assert without_history.data["suggest_stock_amount"] == 84
    assert with_history.data["suggest_stock_amount"] == 84
    assert with_history.data["history_used"] == 0
    assert with_history.data["history_adjusted"] is False


def test_stock_legacy_tuple_contains_workflow_status():
    success = SkillResult(
        success=True,
        data={"daily_avg_sales": 1, "suggest_stock_amount": 8},
    )

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.stock.exec_skill",
            new=AsyncMock(return_value=success),
        ), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.stock.llm_explain",
            new=AsyncMock(return_value=("advice", "template", "")),
        ):
            return await handle_stock(normalize_task_context({"sku": "SKU-1"}))

    ok, error, data = asyncio.run(scenario())
    assert ok is True
    assert error == ""
    assert data["_workflow"]["error_code"] == ""
    assert data["_workflow"]["metadata"]["workflow"] == "stock"
