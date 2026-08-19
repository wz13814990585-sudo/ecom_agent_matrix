"""output_polish：启发式整理 +（可选）DeepSeek 联调。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm.deepseek_client import DeepSeekChatResult
from ecom_agent_matrix.core.llm.output_polish import (
    _extract_existing_answer,
    _heuristic_summary,
    _truncate_json,
    polish_final_output,
)


def test_truncate_json():
    short = _truncate_json({"a": 1}, max_chars=100)
    assert "a" in short
    long = _truncate_json({"x": "y" * 500}, max_chars=80)
    assert "已截断" in long
    assert len(long) <= 80
    print("✅ _truncate_json")


def test_extract_existing_answer():
    assert _extract_existing_answer({"answer": "  你好  "}) == "你好"
    assert _extract_existing_answer({"summary": "摘要"}) == "摘要"
    assert (
        _extract_existing_answer(
            {"react_trace": [{"action": "finish", "final_answer": "文案已生成"}]}
        )
        == "文案已生成"
    )
    # 通用占位不采用
    assert (
        _extract_existing_answer(
            {"react_trace": [{"action": "finish", "final_answer": "任务完成"}]}
        )
        == ""
    )
    assert _extract_existing_answer({}) == ""
    print("✅ _extract_existing_answer")


def test_heuristic_summary():
    s1 = _heuristic_summary(
        success=True,
        data={"gross_profit": 26.79, "profit_ratio": 0.45},
        error_msg="",
        reply_from="calc",
    )
    assert "成功" in s1 and "gross_profit" in s1

    s2 = _heuristic_summary(
        success=False,
        data={
            "sub_results": [
                {"agent": "social_media", "success": True},
                {"agent": "stock", "success": False, "error_msg": "缺 sku"},
            ],
            "working_sku": "SKU-BAG-001",
        },
        error_msg="部分失败",
        reply_from="master_planning",
    )
    assert "失败" in s2
    assert "1/2" in s2
    assert "SKU-BAG-001" in s2
    assert "缺 sku" in s2

    s3 = _heuristic_summary(
        success=True,
        data={"answer": "客服已答复退款流程"},
        error_msg="",
        reply_from="customer_service",
    )
    assert s3 == "客服已答复退款流程"
    print("✅ _heuristic_summary")


async def test_polish_prefer_answer():
    text = await polish_final_output(
        success=True,
        data={"answer": "您好，退款一般 3-7 个工作日。"},
        user_query="退款多久到账",
        reply_from="customer_service",
        prefer_existing_answer=True,
    )
    assert text == "您好，退款一般 3-7 个工作日。"
    print("✅ polish_final_output 复用 answer（不调 LLM）")


async def test_polish_disabled_uses_heuristic():
    old = settings.OUTPUT_POLISH_ENABLED
    settings.OUTPUT_POLISH_ENABLED = False
    try:
        text = await polish_final_output(
            success=True,
            data={"copy_draft": "Hot bag!"},
            reply_from="social",
        )
        assert "copy_draft" in text or "Hot bag" in text
        print("✅ polish 关闭时走启发式")
    finally:
        settings.OUTPUT_POLISH_ENABLED = old


async def test_polish_llm_mocked():
    mock_result = DeepSeekChatResult(
        content="【成功】已为防水背包生成 TikTok 文案。\n· 文案：Hot waterproof bag!",
        mode="chat",
        model="deepseek-chat",
    )
    with patch(
        "ecom_agent_matrix.core.llm.output_polish.deepseek_chat",
        new=AsyncMock(return_value=mock_result),
    ):
        # 临时确保会走 LLM 分支
        old_enabled = settings.OUTPUT_POLISH_ENABLED
        old_key = settings.DEEPSEEK_API_KEY
        settings.OUTPUT_POLISH_ENABLED = True
        settings.DEEPSEEK_API_KEY = old_key or "sk-test"
        try:
            text = await polish_final_output(
                success=True,
                data={
                    "sub_results": [
                        {
                            "agent": "social_media",
                            "success": True,
                            "data": {"copy_draft": "Hot waterproof bag!"},
                        }
                    ],
                    "all_success": True,
                    "timed_out": False,
                    "react_trace": [
                        {"step": 1, "action": "finish", "final_answer": "done"}
                    ],
                },
                user_query="生成tiktok文案",
                reply_from="master_planning",
                prefer_existing_answer=False,
            )
            assert "成功" in text and "文案" in text
            print("✅ polish_final_output Mock LLM")
        finally:
            settings.OUTPUT_POLISH_ENABLED = old_enabled
            settings.DEEPSEEK_API_KEY = old_key


async def test_polish_llm_failure_fallback():
    with patch(
        "ecom_agent_matrix.core.llm.output_polish.deepseek_chat",
        new=AsyncMock(side_effect=RuntimeError("network down")),
    ):
        old_enabled = settings.OUTPUT_POLISH_ENABLED
        old_key = settings.DEEPSEEK_API_KEY
        settings.OUTPUT_POLISH_ENABLED = True
        settings.DEEPSEEK_API_KEY = old_key or "sk-test"
        try:
            text = await polish_final_output(
                success=True,
                data={"gross_profit": 10},
                reply_from="ad",
                prefer_existing_answer=False,
            )
            assert "gross_profit" in text or "成功" in text
            print("✅ LLM 失败时回退启发式")
        finally:
            settings.OUTPUT_POLISH_ENABLED = old_enabled
            settings.DEEPSEEK_API_KEY = old_key


async def test_polish_live_deepseek():
    """有真实 Key 时打一次 DeepSeek；无 Key 则跳过。"""
    if not (settings.DEEPSEEK_API_KEY or "").strip():
        print("⏭ 跳过 live DeepSeek（未配置 DEEPSEEK_API_KEY）")
        return
    if not settings.OUTPUT_POLISH_ENABLED:
        print("⏭ 跳过 live DeepSeek（OUTPUT_POLISH_ENABLED=false）")
        return

    text = await polish_final_output(
        success=True,
        data={
            "sub_results": [
                {
                    "agent": "social_media",
                    "success": True,
                    "data": {
                        "copy_draft": "🔥Hot Sale Waterproof Bag! Limited stock!",
                        "platform": "tiktok",
                    },
                    "error_msg": "",
                }
            ],
            "all_success": True,
            "timed_out": False,
            "working_sku": "SKU-BAG-001",
            "plan": {"planner": "rules", "plan_confidence": 0.8, "agents": ["social_media"]},
            "react_trace": [
                {"step": 1, "action": "call_agent", "thought": "生成社媒文案"},
                {"step": 2, "action": "finish", "final_answer": "文案已生成"},
            ],
        },
        user_query="为防水户外背包生成 tiktok 文案",
        reply_from="master_planning",
        prefer_existing_answer=False,
    )
    assert isinstance(text, str) and len(text) > 5
    print("=== Live DeepSeek 摘要 ===")
    print(text)
    print("✅ polish_final_output live DeepSeek")


async def main():
    test_truncate_json()
    test_extract_existing_answer()
    test_heuristic_summary()
    await test_polish_prefer_answer()
    await test_polish_disabled_uses_heuristic()
    await test_polish_llm_mocked()
    await test_polish_llm_failure_fallback()
    try:
        await test_polish_live_deepseek()
    finally:
        from ecom_agent_matrix.core.llm.deepseek_client import close_http_session

        await close_http_session()
    print("\n全部 output_polish 测试通过")


if __name__ == "__main__":
    asyncio.run(main())
