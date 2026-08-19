"""淘宝开放平台 TOP 工具（由 CRM Agent 按需调用：use_taobao / taobao_method）。"""
from __future__ import annotations

import hashlib
import time
from typing import Any
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill


class TaobaoApiInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    method: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class TaobaoApiOutput(BaseModel):
    """验证外部 JSON 是对象，同时保持旧版扁平返回形状。"""

    model_config = ConfigDict(extra="forbid")
    data: dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def wrap_response(cls, value):
        if isinstance(value, dict) and set(value) != {"data"}:
            return {"data": value}
        return value

    @model_serializer
    def serialize_response(self) -> dict[str, Any]:
        return self.data


def _top_sign(params: dict[str, str], secret: str) -> str:
    """TOP 签名：参数按 key 排序后 secret + key+value... + secret，再 MD5 大写。"""
    pieces = [secret]
    for key in sorted(params):
        pieces.append(f"{key}{params[key]}")
    pieces.append(secret)
    return hashlib.md5("".join(pieces).encode("utf-8")).hexdigest().upper()


@register_skill
class TaobaoApiTool(BaseSkill):
    read_only = False
    side_effect = True
    risk_level = "high"
    timeout_seconds = 30.0
    idempotent = False
    required_scopes = frozenset({"operations:execute"})
    approval_required = True
    input_model = TaobaoApiInput
    output_model = TaobaoApiOutput
    skill_name = "taobao_api"
    skill_desc = (
        "淘宝开放平台接口调用，参数 method=TOP 方法名（如 taobao.trade.fullinfo.get）、"
        "payload=业务参数字典"
    )

    async def run(self, params: dict) -> SkillResult:
        try:
            app_key = (settings.TAOBAO_APP_KEY or "").strip()
            app_secret = (settings.TAOBAO_APP_SECRET or "").strip()
            session_key = (settings.TAOBAO_SESSION_KEY or "").strip()
            api_url = (settings.TAOBAO_API_URL or "").strip() or "https://eco.taobao.com/router/rest"

            if not app_key or not app_secret:
                return SkillResult(
                    success=False,
                    error_msg="未配置 TAOBAO_APP_KEY / TAOBAO_APP_SECRET",
                )

            method = str(params.get("method") or "").strip()
            if not method:
                return SkillResult(success=False, error_msg="缺失参数：method")

            biz = params.get("payload") or {}
            if not isinstance(biz, dict):
                return SkillResult(success=False, error_msg="payload 须为对象")

            sys_params: dict[str, str] = {
                "method": method,
                "app_key": app_key,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "format": "json",
                "v": "2.0",
                "sign_method": "md5",
            }
            if session_key:
                sys_params["session"] = session_key

            for key, value in biz.items():
                if value is None:
                    continue
                sys_params[str(key)] = str(value)

            sys_params["sign"] = _top_sign(sys_params, app_secret)

            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    api_url,
                    content=urlencode(sys_params),
                    headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf-8"},
                )

            if resp.status_code != 200:
                return SkillResult(
                    success=False,
                    error_msg=f"淘宝接口异常，状态码：{resp.status_code}",
                )

            data = resp.json()
            if isinstance(data, dict) and data.get("error_response"):
                err = data["error_response"]
                code = err.get("code") or err.get("sub_code") or "?"
                msg = err.get("msg") or err.get("sub_msg") or str(err)
                return SkillResult(success=False, error_msg=f"淘宝 API 错误 [{code}]: {msg}", data=data)

            return SkillResult(success=True, data=data)
        except Exception as exc:
            return SkillResult(success=False, error_msg=f"接口请求失败：{type(exc).__name__}")
