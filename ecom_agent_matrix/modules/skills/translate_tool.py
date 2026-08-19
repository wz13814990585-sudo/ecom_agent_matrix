"""多语种电商翻译 Skill（LLM）。"""
from __future__ import annotations

import re
from typing import Optional

from ecom_agent_matrix.config.constants import LANG_LIST
from ecom_agent_matrix.core.llm import current_provider_name, llm_chat
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill

SUPPORTED_LANG = set(LANG_LIST)

LANG_NAME = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
}

TRANSLATE_SYSTEM_PROMPT = (
    "You are a professional cross-border ecommerce translator. "
    "Translate product titles, descriptions, and customer-service text accurately. "
    "Return ONLY the translated text. No quotes, no explanation, no language tags."
)


def detect_source_lang(text: str) -> str:
    """简单语种识别：含中文则 zh，否则 en。"""
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def build_translate_prompt(
    text: str,
    target_lang: str,
    source_lang: Optional[str] = None,
) -> str:
    """组装电商翻译 user prompt。"""
    target_name = LANG_NAME.get(target_lang, target_lang)
    source_hint = ""
    if source_lang:
        source_hint = f"Source language: {LANG_NAME.get(source_lang, source_lang)}. "
    return f"{source_hint}Translate the following text into {target_name}:\n{text}"


async def translate_text(
    text: str,
    target_lang: str,
    source_lang: Optional[str] = None,
) -> str:
    """
    电商场景翻译入口：Prompt 拼装 + 调用当前 LLM Provider。
    同语种直接返回原文。
    """
    text = text.strip()
    lang = target_lang.strip().lower()
    if lang not in SUPPORTED_LANG:
        raise ValueError(f"不支持语种：{lang}，仅支持 {sorted(SUPPORTED_LANG)}")

    src = source_lang or detect_source_lang(text)
    if src == lang:
        return text

    return (await llm_chat(
        user_prompt=build_translate_prompt(text, lang, src),
        system_prompt=TRANSLATE_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=256,
        mode="chat",
    )).content


@register_skill
class TranslateTool(BaseSkill):
    skill_name = "text_translate"
    skill_desc = "多语种电商文本翻译（LLM），参数 text / target_lang"

    async def run(self, params: dict) -> SkillResult:
        try:
            text = str(params["text"]).strip()
            lang = str(params["target_lang"]).strip().lower()
            if not text:
                return SkillResult(success=False, error_msg="text 不能为空")
            if lang not in SUPPORTED_LANG:
                return SkillResult(
                    success=False,
                    error_msg=f"不支持语种：{lang}，仅支持 {sorted(SUPPORTED_LANG)}",
                )

            source_lang = detect_source_lang(text)
            trans_text = await translate_text(text, lang, source_lang)

            return SkillResult(
                success=True,
                data={
                    "provider": current_provider_name(),
                    "source_lang": source_lang,
                    "target_lang": lang,
                    "source_text": text,
                    "trans_text": trans_text,
                },
            )
        except KeyError as e:
            return SkillResult(success=False, error_msg=f"缺少参数：{e}")
        except ValueError as e:
            return SkillResult(success=False, error_msg=str(e))
        except Exception as e:
            return SkillResult(success=False, error_msg=f"翻译失败：{e}")
