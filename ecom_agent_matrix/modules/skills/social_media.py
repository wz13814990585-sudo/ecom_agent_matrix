"""社媒文案生成 Skill（LLM，无 Key 时模板兜底）。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ecom_agent_matrix.config.constants import LANG_LIST
from ecom_agent_matrix.core.llm import current_provider_name, is_llm_configured, llm_chat
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill
from ecom_agent_matrix.modules.parsers.social import SUPPORTED_PLATFORMS

LANG_NAME = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
}


class SocialMediaCopyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    product_name: str = Field(min_length=1)
    feature: str = Field(min_length=1)
    platform: Literal["tiktok", "instagram", "facebook", "twitter", "youtube"]
    lang: str = Field(default="en", min_length=1)


class SocialMediaCopyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    copy_draft: str
    platform: str
    lang: str
    source: str
    llm_error: str

COPY_SYSTEM_PROMPT = """You are a cross-border ecommerce social media copywriter.
Write ONE short promotional caption for the given platform and language.
Rules:
- Return ONLY the caption text, no quotes, no markdown, no explanation
- Match platform tone (TikTok energetic, Instagram aesthetic, Facebook clear CTA, Twitter concise, YouTube hook)
- Include the product name and key selling point naturally
- Keep it under 280 characters unless platform is youtube (then under 400)
"""


def _template_copy(product_name: str, feature: str, platform: str, lang: str) -> str:
    template_map = {
        "tiktok": f"🔥Hot Sale {product_name}! {feature} Limited stock now!",
        "instagram": f"New Arrival ✨ {product_name} | {feature}",
        "facebook": f"Discover {product_name} — {feature}. Shop now!",
        "twitter": f"{product_name}: {feature} #Sale",
        "youtube": f"Unboxing {product_name}! {feature}",
    }
    draft = template_map[platform]
    if lang == "zh":
        return f"【{platform}】{product_name}｜卖点：{feature}｜限时热卖！"
    return draft


@register_skill
class SocialMediaCopyTool(BaseSkill):
    read_only = True
    side_effect = False
    risk_level = "low"
    timeout_seconds = 45.0
    idempotent = False
    input_model = SocialMediaCopyInput
    output_model = SocialMediaCopyOutput
    skill_name = "social_media_gen"
    skill_desc = (
        "社媒带货文案生成（LLM），参数 product_name、feature、platform"
        f"（支持 {', '.join(sorted(SUPPORTED_PLATFORMS))}）、lang"
    )

    async def run(self, params: dict) -> SkillResult:
        try:
            product_name = str(params["product_name"]).strip()
            feature = str(params["feature"]).strip()
            platform = str(params["platform"]).strip().lower()
            lang = str(params.get("lang") or "en").strip().lower()

            if not product_name:
                return SkillResult(success=False, error_msg="product_name 为空")
            if platform not in SUPPORTED_PLATFORMS:
                return SkillResult(
                    success=False,
                    error_msg=(
                        f"不支持的平台：{platform}，"
                        f"可选：{', '.join(sorted(SUPPORTED_PLATFORMS))}"
                    ),
                )
            if lang not in LANG_LIST:
                lang = "en"

            source = "template"
            llm_error = ""
            draft = _template_copy(product_name, feature, platform, lang)
            if is_llm_configured():
                try:
                    lang_name = LANG_NAME.get(lang, lang)
                    user_prompt = (
                        f"Platform: {platform}\n"
                        f"Language: {lang_name} ({lang})\n"
                        f"Product: {product_name}\n"
                        f"Selling point: {feature}\n"
                        "Write the caption now."
                    )
                    text = (
                        await llm_chat(
                            user_prompt=user_prompt,
                            system_prompt=COPY_SYSTEM_PROMPT,
                            temperature=0.7,
                            max_tokens=300,
                            mode="chat",
                        )
                    ).content.strip()
                    if text:
                        draft = text
                        source = current_provider_name()
                    else:
                        llm_error = "empty_content"
                except Exception as exc:
                    llm_error = type(exc).__name__

            return SkillResult(
                success=True,
                data={
                    "copy_draft": draft.strip(),
                    "platform": platform,
                    "lang": lang,
                    "source": source,
                    "llm_error": llm_error,
                },
            )
        except KeyError as e:
            return SkillResult(success=False, error_msg=f"缺失参数：{e}")
        except Exception as exc:
            return SkillResult(success=False, error_msg=f"社媒文案生成失败：{type(exc).__name__}")
