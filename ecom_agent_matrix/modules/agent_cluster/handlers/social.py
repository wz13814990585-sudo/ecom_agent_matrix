"""社媒营销 workflow：typed request + 两个 Skill 并行编排。"""
from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError
from ecom_agent_matrix.platform.observability.metrics import observed_workflow

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.rate_limit import acquire_slot
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.core.tasking import TaskContext, WorkflowResult, ensure_task_context
from ecom_agent_matrix.core.tasking.result import (
    INVALID_REQUEST,
    MISSING_PRODUCT,
    PARTIAL_SUCCESS,
    SKILL_FAILED,
    UNSUPPORTED_PLATFORM,
    WORKFLOW_TIMEOUT,
)
from ecom_agent_matrix.modules.parsers.social import (
    UnsupportedSocialPlatform,
    normalize_social_platform,
    parse_social_request,
)
from ecom_agent_matrix.modules.parsers.social import SUPPORTED_PLATFORMS

# 兼容此前可能使用的 public helper 名称；解析实现位于 domain parser。
normalize_platform = normalize_social_platform


def _metadata(started: float, **extra) -> dict:
    return {
        "workflow": "social",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        **extra,
    }


@observed_workflow("social")
async def run_social_workflow(task: dict | TaskContext) -> WorkflowResult:
    """解析请求，在 workflow deadline 内并发生成文案和绘图提示词。"""
    started = time.perf_counter()
    ctx = ensure_task_context(task)
    try:
        request = parse_social_request(ctx)
    except UnsupportedSocialPlatform as exc:
        return WorkflowResult(
            success=False,
            error_code=UNSUPPORTED_PLATFORM,
            error_msg=(
                f"不支持的平台：{exc.platform}，"
                f"可选：{', '.join(sorted(SUPPORTED_PLATFORMS))}"
            ),
            data={
                "exec_kind": "social",
                "product_name": ctx.product_name or "",
                "platform": exc.platform,
                "supported_platforms": sorted(SUPPORTED_PLATFORMS),
            },
            metadata=_metadata(started),
        )
    except (ValidationError, TypeError, ValueError) as exc:
        return WorkflowResult(
            success=False,
            error_code=INVALID_REQUEST,
            error_msg=f"社媒请求参数不合法：{exc}",
            data={"exec_kind": "social"},
            metadata=_metadata(started),
        )

    if not request.product_name:
        return WorkflowResult(
            success=False,
            error_code=MISSING_PRODUCT,
            error_msg="缺少 product_name：请在 query 中写明「为「商品名」生成文案」",
            data={
                "exec_kind": "social",
                "product_name": "",
                "platform": request.platform,
            },
            metadata=_metadata(started),
        )

    workflow_timeout = float(settings.SOCIAL_SKILL_TIMEOUT)
    async with acquire_slot(
        "social",
        limit=int(settings.SOCIAL_MAX_CONCURRENT),
        mode=settings.SOCIAL_RATE_LIMIT_MODE,
        ttl_sec=max(workflow_timeout + 5.0, 30.0),
    ) as rate_backend:
        try:
            copy_res, prompt_res = await asyncio.wait_for(
                asyncio.gather(
                    exec_skill(
                        "social_media_gen",
                        {
                            "product_name": request.product_name,
                            "feature": request.feature,
                            "platform": request.platform,
                            "lang": request.lang,
                        },
                    ),
                    exec_skill(
                        "ai_prompt_generate",
                        {
                            "product": request.product_name,
                            "scene": request.scene,
                            "style": request.style,
                        },
                    ),
                    return_exceptions=True,
                ),
                timeout=workflow_timeout,
            )
        except asyncio.TimeoutError:
            error_msg = f"workflow 调用超时（>{workflow_timeout}s）"
            return WorkflowResult(
                success=False,
                error_code=WORKFLOW_TIMEOUT,
                error_msg=error_msg,
                data={
                    "exec_kind": "social",
                    "product_name": request.product_name,
                    "platform": request.platform,
                    "skill_status": {
                        "social_media_gen": False,
                        "ai_prompt_generate": False,
                    },
                    "skill_errors": {"timeout": error_msg},
                    "partial_success": False,
                    "rate_limit_backend": rate_backend,
                },
                metadata=_metadata(started),
            )

        results = {
            "social_media_gen": copy_res,
            "ai_prompt_generate": prompt_res,
        }
        skill_errors: dict[str, str] = {}
        skill_error_codes: dict[str, str] = {}
        skill_status: dict[str, bool] = {}
        output: dict[str, dict] = {
            "social_media_gen": {},
            "ai_prompt_generate": {},
        }
        for skill_name, result in results.items():
            if isinstance(result, BaseException):
                skill_status[skill_name] = False
                skill_errors[skill_name] = type(result).__name__
                skill_error_codes[skill_name] = SKILL_FAILED
                continue
            skill_status[skill_name] = bool(result.success)
            if result.success:
                output[skill_name] = result.data or {}
            else:
                skill_errors[skill_name] = result.error_msg or "失败"
                skill_error_codes[skill_name] = result.error_code or SKILL_FAILED

        success_count = sum(skill_status.values())
        partial = success_count == 1
        success = success_count > 0
        error_code = PARTIAL_SUCCESS if partial else ("" if success else SKILL_FAILED)
        error_msg = "; ".join(f"{key}: {value}" for key, value in skill_errors.items())
        return WorkflowResult(
            success=success,
            partial_success=partial,
            error_code=error_code,
            error_msg=error_msg,
            data={
                "exec_kind": "social",
                "product_name": request.product_name,
                "feature": request.feature,
                "platform": request.platform,
                "lang": request.lang,
                "scene": request.scene,
                "style": request.style,
                "social_copy": output["social_media_gen"],
                "ai_image_prompt": output["ai_prompt_generate"],
                "skill_status": skill_status,
                "skill_errors": skill_errors,
                "partial_success": partial,
                "rate_limit_backend": rate_backend,
            },
            metadata=_metadata(started, skill_error_codes=skill_error_codes),
        )


async def handle_social(task: dict | TaskContext) -> tuple[bool, str, dict]:
    """兼容旧 Exec Agent/Handler tuple 协议。"""
    return (await run_social_workflow(task)).as_legacy_tuple()
