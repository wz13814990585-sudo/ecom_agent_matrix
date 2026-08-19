"""AI 绘图提示词 Skill（LLM，无 Key 时模板兜底）。"""
from __future__ import annotations

from ecom_agent_matrix.core.llm import current_provider_name, is_llm_configured, llm_chat
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill

PROMPT_SYSTEM = """You are an expert ecommerce product photography prompt engineer for AI image models.
Return ONLY one English positive prompt suitable for Stable Diffusion / Midjourney style models.
Rules:
- Include product, scene, lighting, style, camera details
- No quotation marks, no markdown, no explanation
- Single paragraph, under 80 words
"""


def _template_prompt(product: str, scene: str, style: str) -> str:
    return (
        f"Product photography of {product}, scene:{scene}, style:{style}, "
        "8k, ultra realistic, soft natural light"
    )


@register_skill
class AIPromptGenTool(BaseSkill):
    skill_name = "ai_prompt_generate"
    skill_desc = "商品 AI 绘图提示词生成（LLM），参数 product、scene、style"

    async def run(self, params: dict) -> SkillResult:
        try:
            product = str(params["product"]).strip()
            scene = str(params.get("scene") or "e-commerce product shooting").strip()
            style = str(params.get("style") or "bright commercial").strip()

            if not product:
                return SkillResult(success=False, error_msg="product 为空")

            source = "template"
            llm_error = ""
            prompt = _template_prompt(product, scene, style)
            if is_llm_configured():
                try:
                    user_prompt = (
                        f"Product: {product}\n"
                        f"Scene: {scene}\n"
                        f"Style: {style}\n"
                        "Generate the positive prompt now."
                    )
                    text = (
                        await llm_chat(
                            user_prompt=user_prompt,
                            system_prompt=PROMPT_SYSTEM,
                            temperature=0.5,
                            max_tokens=200,
                            mode="chat",
                        )
                    ).content.strip()
                    if text:
                        prompt = text
                        source = current_provider_name()
                    else:
                        llm_error = "empty_content"
                except Exception as exc:
                    llm_error = str(exc)

            return SkillResult(
                success=True,
                data={
                    "positive_prompt": prompt.strip(),
                    "source": source,
                    "llm_error": llm_error,
                },
            )
        except KeyError as err:
            return SkillResult(success=False, error_msg=f"缺失参数：{err}")
        except Exception as exc:
            return SkillResult(success=False, error_msg=f"绘图提示词生成失败：{exc}")
