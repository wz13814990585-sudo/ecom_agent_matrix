"""客服答复 handler：翻译 / 淘宝查单编排后生成回复。由 Exec Agent 调用，不是独立 Agent。"""
from __future__ import annotations

import re

from ecom_agent_matrix.config.constants import LANG_LIST
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.memory.short_memory import AgentShortMemory
from ecom_agent_matrix.core.skill.skill_registry import exec_skill

logger = setup_logger("agent.crm")

_ORDER_PATTERN = re.compile(
    r"\b(?:ORD[-_][A-Z0-9_-]+|\d{10,20})\b",
    re.IGNORECASE,
)
_CJK = re.compile(r"[\u4e00-\u9fff]")


def _infer_lang(payload: dict, user_query: str) -> str:
    explicit = str(payload.get("lang") or "").strip().lower()
    if explicit in LANG_LIST:
        return explicit
    return "zh" if _CJK.search(user_query or "") else "en"


async def _maybe_taobao(payload: dict) -> dict:
    method = str(payload.get("taobao_method") or "").strip()
    taobao_payload = payload.get("taobao_payload")
    if not isinstance(taobao_payload, dict):
        taobao_payload = {}

    if not method and payload.get("use_taobao"):
        order_no = str(payload.get("order_no") or "").strip()
        if not order_no:
            m = _ORDER_PATTERN.search(str(payload.get("query") or payload.get("user_query") or ""))
            order_no = m.group(0) if m else ""
        if order_no:
            method = "taobao.trade.fullinfo.get"
            taobao_payload = {"tid": order_no, "fields": "tid,status,total_fee,payment,orders"}

    if not method:
        return {"skipped": True}

    res = await exec_skill(
        "taobao_api",
        {"method": method, "payload": taobao_payload},
    )
    return {
        "skipped": False,
        "success": res.success,
        "error_msg": res.error_msg,
        "data": res.data or {},
        "method": method,
    }


async def handle_crm(payload: dict, *, task_id: str = "") -> tuple[bool, str, dict]:
    """多语种客服答复（薄编排）。"""
    session_id = str(payload.get("session_id") or task_id)
    user_query = str(payload.get("user_query") or payload.get("query", "")).strip()
    user_lang = _infer_lang(payload, user_query)
    is_fallback_route = bool(payload.get("_fallback_route"))

    if not user_query:
        return False, "user_query 为空", {"exec_kind": "crm", "session_id": session_id}

    short_mem = AgentShortMemory(session_id=session_id)
    await short_mem.append(role="user", content=user_query)
    history = await short_mem.get_all()

    translate_ok = True
    translate_error = ""
    trans_data: dict = {}
    if user_lang != "zh":
        trans_res = await exec_skill(
            "text_translate",
            {"text": user_query, "target_lang": user_lang},
        )
        translate_ok = bool(trans_res.success)
        translate_error = trans_res.error_msg or ""
        trans_data = trans_res.data if trans_res.success else {}

    taobao_info = await _maybe_taobao(payload)

    reply_res = await exec_skill(
        "crm_reply",
        {
            "user_query": user_query,
            "lang": user_lang,
            "history": history or [],
            "use_rag": payload.get("use_rag"),
            "taobao_info": taobao_info,
            "is_fallback_route": is_fallback_route,
            "task_id": task_id,
        },
    )
    reply_data = reply_res.data or {}
    answer = str(reply_data.get("answer") or "")
    llm_ok = bool(reply_data.get("llm_ok"))
    rag_used = bool(reply_data.get("rag_used"))
    rag_doc_count = int(reply_data.get("rag_doc_count") or 0)
    rag_error = str(reply_data.get("rag_error") or "")

    if not reply_res.success:
        answer = answer or "客服答复服务暂不可用，请稍后再试。"
        rag_error = rag_error or (reply_res.error_msg or "crm_reply failed")

    await short_mem.append(role="assistant", content=answer)

    errors: list[str] = []
    if not translate_ok and translate_error:
        errors.append(f"translate: {translate_error}")
    if rag_error:
        errors.append(f"rag: {rag_error}")
    if not taobao_info.get("skipped") and not taobao_info.get("success"):
        errors.append(f"taobao: {taobao_info.get('error_msg') or 'failed'}")
    if not reply_res.success:
        errors.append(f"crm_reply: {reply_res.error_msg or 'failed'}")
    if (
        not llm_ok
        and settings.DEEPSEEK_API_KEY
        and not is_fallback_route
        and not rag_used
    ):
        errors.append("llm_unavailable_used_fallback")

    return (
        True,
        "; ".join(errors),
        {
            "exec_kind": "crm",
            "session_id": session_id,
            "answer": answer,
            "lang": user_lang,
            "llm_ok": llm_ok,
            "translate_ok": translate_ok,
            "trans_info": trans_data,
            "rag_used": rag_used,
            "rag_doc_count": rag_doc_count,
            "taobao": taobao_info,
            "partial_success": bool(errors),
        },
    )
