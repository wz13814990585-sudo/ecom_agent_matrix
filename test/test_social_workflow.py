"""Phase 2C-1 Social parser / workflow 测试。"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import patch

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.skill.base_skill import SkillResult
from ecom_agent_matrix.core.tasking import normalize_task_context
from ecom_agent_matrix.core.tasking.result import (
    MISSING_PRODUCT,
    PARTIAL_SUCCESS,
    SKILL_FAILED,
    UNSUPPORTED_PLATFORM,
    WORKFLOW_TIMEOUT,
)
from ecom_agent_matrix.modules.agent_cluster.handlers import social as social_handler
from ecom_agent_matrix.modules.agent_cluster.handlers.social import (
    handle_social,
    run_social_workflow,
)
from ecom_agent_matrix.modules.parsers.social import parse_social_request


def _success_for(skill_name: str) -> SkillResult:
    if skill_name == "social_media_gen":
        return SkillResult(success=True, data={"copy_draft": "caption"})
    return SkillResult(success=True, data={"positive_prompt": "image prompt"})


def test_social_explicit_platform_wins_over_query():
    ctx = normalize_task_context(
        {
            "query": "为「防水背包」生成 TikTok 文案",
            "platform": "instagram",
        }
    )
    assert parse_social_request(ctx).platform == "instagram"


def test_social_tiktok_and_instagram_aliases_are_normalized():
    tiktok = parse_social_request(
        normalize_task_context({"query": "为「背包」生成抖音文案"})
    )
    instagram = parse_social_request(
        normalize_task_context({"query": "为「背包」生成 IG 文案"})
    )
    assert tiktok.platform == "tiktok"
    assert instagram.platform == "instagram"


def test_social_short_platform_aliases_do_not_match_inside_words():
    request = parse_social_request(
        normalize_task_context(
            {"query": "为「storage box」生成文案", "product_name": "storage box"}
        )
    )
    assert request.platform == "tiktok"


def test_social_unsupported_platform_has_structured_error():
    result = asyncio.run(
        run_social_workflow(
            {"product_name": "防水背包", "platform": "xiaohongshu"}
        )
    )
    assert result.success is False
    assert result.error_code == UNSUPPORTED_PLATFORM
    assert result.data["platform"] == "xiaohongshu"


def test_social_explicit_product_name_wins_over_query():
    request = parse_social_request(
        normalize_task_context(
            {
                "query": "为「鞋子」生成 TikTok 文案",
                "product_name": "防水背包",
            }
        )
    )
    assert request.product_name == "防水背包"


def test_social_conservative_product_extraction_requests_clarification():
    ctx = normalize_task_context({"query": "帮我生成 TikTok 文案，顺便看看库存"})
    assert parse_social_request(ctx).product_name is None
    result = asyncio.run(run_social_workflow(ctx))
    assert result.success is False
    assert result.error_code == MISSING_PRODUCT


def test_social_feature_and_language_are_parsed():
    request = parse_social_request(
        normalize_task_context(
            {
                "query": "为「防水背包」生成 IG 文案，卖点：轻便防水",
                "language": "ZH",
            }
        )
    )
    assert request.product_name == "防水背包"
    assert request.feature == "轻便防水"
    assert request.lang == "zh"


def test_social_parser_does_not_reread_conflicting_user_query():
    request = parse_social_request(
        normalize_task_context(
            {
                "query": "为「防水背包」生成 IG 文案",
                "user_query": "为「鞋子」生成 TikTok 文案",
            }
        )
    )
    assert request.product_name == "防水背包"
    assert request.platform == "instagram"


def test_social_skills_run_concurrently_and_receive_typed_params():
    entered: list[str] = []
    calls: dict[str, dict] = {}

    async def concurrent_skill(skill_name: str, params: dict):
        entered.append(skill_name)
        calls[skill_name] = params
        if len(entered) == 1:
            for _ in range(20):
                if len(entered) == 2:
                    break
                await asyncio.sleep(0)
        assert len(entered) == 2
        return _success_for(skill_name)

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.social.exec_skill",
            side_effect=concurrent_skill,
        ):
            return await run_social_workflow(
                {
                    "query": "为「防水背包」生成 IG 文案",
                    "feature": "防水",
                    "lang": "zh",
                    "scene": "outdoor",
                    "style": "cinematic",
                }
            )

    result = asyncio.run(scenario())
    assert result.success is True
    assert result.partial_success is False
    assert calls["social_media_gen"] == {
        "product_name": "防水背包",
        "feature": "防水",
        "platform": "instagram",
        "lang": "zh",
    }
    assert calls["ai_prompt_generate"] == {
        "product": "防水背包",
        "scene": "outdoor",
        "style": "cinematic",
    }


def test_social_one_skill_failure_is_partial_success():
    async def skill_result(skill_name: str, params: dict):
        if skill_name == "social_media_gen":
            return SkillResult(
                success=False,
                error_code="TIMEOUT",
                error_msg="copy timeout",
            )
        return _success_for(skill_name)

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.social.exec_skill",
            side_effect=skill_result,
        ):
            return await run_social_workflow(
                {"product_name": "防水背包", "platform": "tiktok"}
            )

    result = asyncio.run(scenario())
    assert result.success is True
    assert result.partial_success is True
    assert result.error_code == PARTIAL_SUCCESS
    assert result.metadata["skill_error_codes"]["social_media_gen"] == "TIMEOUT"


def test_social_two_skill_failures_fail_workflow():
    async def failed(skill_name: str, params: dict):
        return SkillResult(
            success=False,
            error_code="EXECUTION_ERROR",
            error_msg=f"{skill_name} failed",
        )

    async def scenario():
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.social.exec_skill",
            side_effect=failed,
        ):
            return await run_social_workflow({"product_name": "防水背包"})

    result = asyncio.run(scenario())
    assert result.success is False
    assert result.partial_success is False
    assert result.error_code == SKILL_FAILED
    assert set(result.metadata["skill_error_codes"]) == {
        "social_media_gen",
        "ai_prompt_generate",
    }


def test_social_workflow_deadline_is_structured():
    async def blocked(skill_name: str, params: dict):
        await asyncio.sleep(1)
        return _success_for(skill_name)

    async def scenario():
        with patch.object(settings, "SOCIAL_SKILL_TIMEOUT", 0.01), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.social.exec_skill",
            side_effect=blocked,
        ):
            return await run_social_workflow({"product_name": "防水背包"})

    result = asyncio.run(scenario())
    assert result.success is False
    assert result.error_code == WORKFLOW_TIMEOUT
    assert result.metadata["workflow"] == "social"


def test_social_legacy_handler_accepts_context_and_returns_tuple():
    async def successful(skill_name: str, params: dict):
        return _success_for(skill_name)

    async def scenario():
        ctx = normalize_task_context({"product_name": "防水背包"})
        with patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.social.exec_skill",
            side_effect=successful,
        ):
            return await handle_social(ctx)

    legacy = asyncio.run(scenario())
    assert isinstance(legacy, tuple)
    assert legacy[0] is True
    assert legacy[2]["partial_success"] is False


def test_social_handler_contains_no_core_regex_parser():
    source = inspect.getsource(social_handler)
    assert "re.compile" not in source
    assert "_PLATFORM_PATTERN" not in source
    assert "def extract_product_name" not in source
    assert "def extract_feature" not in source
