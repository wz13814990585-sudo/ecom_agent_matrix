"""客服答复 Skill：RAG 拼装 + LLM 生成 + 兜底文案（无 MCP 依赖）。"""
from __future__ import annotations

import re
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ecom_agent_matrix.config.constants import LANG_LIST
from ecom_agent_matrix.core.llm import is_llm_configured, llm_chat
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill

CRM_SYSTEM_PROMPT = (
    "你是跨境独立站电商客服助手。用简洁、礼貌的中文或用户指定语种回答。"
    "可处理退款、物流、订单状态、商品咨询。"
    "若下方提供了「商品知识检索」且非空：必须优先依据检索结果作答；"
    "可引用其中的 SKU / 商品名；不要因为用户未给链接就拒绝回答。"
    "若检索到多个候选商品，列出最相关的 1~3 个并请用户确认。"
    "仅当检索结果明确缺失某细节（如清洗步骤）时，才说明「知识库暂无该细节」，"
    "可基于已有卖点做有限合理说明，并邀请用户补充 SKU/订单号。"
    "不要编造物流单号或退款到账时间。"
    "若提供「已验证业务上下文」，优先使用其中订单事实与政策内容，不要编造缺失字段。"
)

_KNOWLEDGE_HINT = re.compile(
    r"(材质|规格|怎么用|如何使用|介绍|知识|防水|尺寸|清洗|保养|面料|成分|"
    r"背包|商品|款式|faq|how to|what is|material|care|wash|fabric)",
    re.IGNORECASE,
)


class CrmReplyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_query: str | None = None
    query: str | None = None
    lang: str = "zh"
    history: list[dict[str, Any]] = Field(default_factory=list)
    use_rag: bool | None = None
    taobao_info: dict[str, Any] = Field(default_factory=dict)
    is_fallback_route: bool = False
    task_id: str | None = None
    upstream_context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_query(self) -> "CrmReplyInput":
        if not (self.user_query or self.query):
            raise ValueError("user_query 为空")
        return self


class CrmReplyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    llm_ok: bool
    rag_used: bool
    rag_doc_count: int = Field(ge=0)
    rag_error: str
    llm_error: str = ""


def should_use_rag(user_query: str, use_rag_flag) -> bool:
    """use_rag_flag: True/False 显式；None 则按问题启发式。"""
    if use_rag_flag is False:
        return False
    if use_rag_flag is True:
        return True
    return bool(_KNOWLEDGE_HINT.search(user_query or ""))


def format_rag_docs(docs: list[dict], limit: int = 3) -> str:
    lines: list[str] = []
    for i, doc in enumerate(docs[:limit], 1):
        sku = doc.get("sku") or doc.get("goods_sku") or ""
        title = doc.get("title") or doc.get("product_name") or sku or f"doc-{i}"
        body = (
            doc.get("chunk_text")
            or doc.get("content")
            or doc.get("text")
            or doc.get("chunk")
            or ""
        )
        snippet = str(body).strip().replace("\n", " ")[:320]
        if not snippet:
            continue
        label = f"{title}" if not sku or sku in str(title) else f"{title} | {sku}"
        lines.append(f"{i}. [{label}] {snippet}")
    return "\n".join(lines)


def fallback_answer(user_query: str, *, is_fallback_route: bool) -> str:
    if is_fallback_route:
        return (
            "抱歉，我暂时无法识别该请求的完整意图。"
            "请描述您的订单、退款或商品相关问题，我会尽力帮助您。"
        )
    return (
        f"已收到您的问题：{user_query}。"
        "当前大模型未配置或不可用，请稍后再试，或提供订单号以便人工跟进。"
    )


async def retrieve_rag_docs(
    user_query: str, lang: str, task_id: str | None
) -> tuple[list[dict], str]:
    try:
        from ecom_agent_matrix.modules.rag.retriever import hybrid_retrieve

        docs, _cached, _ms = await hybrid_retrieve(
            user_query,
            lang if lang in LANG_LIST else "en",
            None,
            5,
            task_id=task_id,
        )
        return list(docs or []), ""
    except Exception as exc:
        return [], type(exc).__name__


@register_skill
class CrmReplyTool(BaseSkill):
    read_only = True
    side_effect = False
    risk_level = "low"
    timeout_seconds = 60.0
    idempotent = False
    input_model = CrmReplyInput
    output_model = CrmReplyOutput
    skill_name = "crm_reply"
    skill_desc = (
        "客服答复生成：参数 user_query、lang、history、use_rag、taobao_info、"
        "is_fallback_route、task_id"
    )

    async def run(self, params: dict) -> SkillResult:
        try:
            user_query = str(params.get("user_query") or params.get("query") or "").strip()
            if not user_query:
                return SkillResult(success=False, error_msg="user_query 为空")

            lang = str(params.get("lang") or "zh").strip().lower()
            if lang not in LANG_LIST:
                lang = "zh"

            history = params.get("history") or []
            if not isinstance(history, list):
                history = []

            is_fallback_route = bool(params.get("is_fallback_route"))
            use_rag_flag = params.get("use_rag")  # True/False/None
            task_id = params.get("task_id")
            taobao_info = params.get("taobao_info")
            if not isinstance(taobao_info, dict):
                taobao_info = {"skipped": True}
            upstream_context = params.get("upstream_context") or {}
            if not isinstance(upstream_context, dict):
                upstream_context = {}

            rag_docs: list[dict] = []
            rag_error = ""
            rag_used = False
            if should_use_rag(user_query, use_rag_flag) and not is_fallback_route:
                rag_docs, rag_error = await retrieve_rag_docs(
                    user_query, lang, str(task_id) if task_id else None
                )
                rag_used = bool(rag_docs)

            answer_text = ""
            llm_ok = False
            if is_llm_configured() and not is_fallback_route:
                try:
                    hist_snip = "\n".join(
                        f"{h.get('role', '?')}: {h.get('content', '')}"
                        for h in history[-6:]
                        if isinstance(h, dict)
                    )
                    rag_block = format_rag_docs(rag_docs)
                    verified_context = json.dumps(
                        upstream_context,
                        ensure_ascii=False,
                        default=str,
                    )[:3500]
                    taobao_block = ""
                    if not taobao_info.get("skipped"):
                        taobao_block = (
                            f"\n淘宝查询({taobao_info.get('method')}): "
                            f"{'成功' if taobao_info.get('success') else taobao_info.get('error_msg')}\n"
                            f"{str(taobao_info.get('data') or '')[:800]}"
                        )
                    answer = await llm_chat(
                        user_prompt=(
                            f"用户语种偏好: {lang}\n"
                            f"近期对话:\n{hist_snip}\n\n"
                            f"商品知识检索:\n{rag_block or '(无)'}\n"
                            f"已验证业务上下文:\n{verified_context or '(无)'}\n"
                            f"{taobao_block}\n"
                            f"当前问题: {user_query}"
                        ),
                        system_prompt=CRM_SYSTEM_PROMPT,
                        temperature=0.3,
                        max_tokens=600,
                        mode="chat",
                    )
                    llm_ok = bool(answer.content.strip())
                    answer_text = answer.content
                except Exception as exc:
                    return SkillResult(
                        success=True,
                        data={
                            "answer": fallback_answer(
                                user_query, is_fallback_route=is_fallback_route
                            )
                            if not rag_used
                            else (
                                f"根据商品知识库，与「{user_query}」相关的要点如下：\n"
                                f"{format_rag_docs(rag_docs)}\n"
                                "如需更具体答复请补充订单号或商品 SKU。"
                            ),
                            "llm_ok": False,
                            "rag_used": rag_used,
                            "rag_doc_count": len(rag_docs),
                            "rag_error": rag_error or type(exc).__name__,
                            "llm_error": type(exc).__name__,
                        },
                    )

            if not llm_ok:
                if rag_used:
                    answer_text = (
                        f"根据商品知识库，与「{user_query}」相关的要点如下：\n"
                        f"{format_rag_docs(rag_docs)}\n"
                        "如需更具体答复请补充订单号或商品 SKU。"
                    )
                else:
                    answer_text = fallback_answer(user_query, is_fallback_route=is_fallback_route)

            return SkillResult(
                success=True,
                data={
                    "answer": answer_text,
                    "llm_ok": llm_ok,
                    "rag_used": rag_used,
                    "rag_doc_count": len(rag_docs),
                    "rag_error": rag_error,
                },
            )
        except Exception as exc:
            return SkillResult(success=False, error_msg=f"客服答复失败：{type(exc).__name__}")
