"""LLM 解读层与报表结构化摘要单测（mock，不打真实 API）。"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.modules.skills.ops_report import (
    _format_structured_summary,
    _template_summary,
)
from ecom_agent_matrix.modules.utils.llm_explain import llm_explain


def test_ops_report_template_has_actions():
    sections = {
        "sales": {
            "days": 7,
            "order_count": 10,
            "units_sold": 20,
            "gmv": 100.0,
            "refund_orders": 2,
            "refund_rate": 0.2,
        },
        "stock": {
            "sku_count": 5,
            "total_stock": 100,
            "out_of_stock_skus": 1,
            "low_stock_skus": 2,
        },
    }
    text = _template_summary("daily_ops", sections)
    assert "异常点" in text and "建议动作" in text


def test_format_structured_summary():
    out = _format_structured_summary(
        {
            "summary": "本周需关注退款与缺货。",
            "anomalies": ["退款率偏高"],
            "hypotheses": ["尺码表不清"],
            "actions": ["核对退款原因", "补货缺货 SKU", "复盘详情页"],
        },
        "fallback",
    )
    assert "退款率偏高" in out and "1) 核对退款原因" in out


async def test_llm_explain_fallback_without_key():
    old_key = settings.DEEPSEEK_API_KEY
    old_flag = settings.AGENT_LLM_EXPLAIN_ENABLED
    settings.DEEPSEEK_API_KEY = ""
    settings.AGENT_LLM_EXPLAIN_ENABLED = True
    try:
        text, source, err = await llm_explain(
            system_prompt="sys",
            user_prompt="user",
            fallback="兜底文案",
        )
        assert text == "兜底文案" and source == "template" and err == "no_api_key"
    finally:
        settings.DEEPSEEK_API_KEY = old_key
        settings.AGENT_LLM_EXPLAIN_ENABLED = old_flag


async def test_llm_explain_uses_deepseek():
    from ecom_agent_matrix.core.llm.deepseek_client import DeepSeekChatResult

    old_key = settings.DEEPSEEK_API_KEY
    old_flag = settings.AGENT_LLM_EXPLAIN_ENABLED
    settings.DEEPSEEK_API_KEY = "sk-test"
    settings.AGENT_LLM_EXPLAIN_ENABLED = True
    try:
        with patch(
            "ecom_agent_matrix.modules.utils.llm_explain.deepseek_chat",
            new=AsyncMock(return_value=DeepSeekChatResult(content="LLM 解读")),
        ):
            text, source, err = await llm_explain(
                system_prompt="sys",
                user_prompt="user",
                fallback="兜底",
            )
        assert text == "LLM 解读" and source == "deepseek" and err == ""
    finally:
        settings.DEEPSEEK_API_KEY = old_key
        settings.AGENT_LLM_EXPLAIN_ENABLED = old_flag


if __name__ == "__main__":
    test_ops_report_template_has_actions()
    test_format_structured_summary()
    asyncio.run(test_llm_explain_fallback_without_key())
    asyncio.run(test_llm_explain_uses_deepseek())
    print("✅ llm_explain / ops_report 结构测试通过")
