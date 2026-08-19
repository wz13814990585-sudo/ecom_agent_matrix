"""最终结果整理：用 LLM 把 Agent/Skill 原始 JSON 写成可读中文摘要。"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ecom_agent_matrix.core.llm.types import ChatResult

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm.router import is_llm_configured, llm_chat
from ecom_agent_matrix.core.logging_config import setup_logger

logger = setup_logger("llm.output_polish")

_SYSTEM_PROMPT = """你是跨境电商多智能体系统的「结果整理」助手。
把系统返回的原始结构化结果，整理成运营人员一眼能看懂的中文说明。

硬性要求：
1. 开头用一句话给出结论（成功 / 失败 / 部分成功）
2. 用短条目列出关键数字、结论与建议；不要复述整段 JSON 字段名堆砌
3. 有错误时说明原因；不要编造原始结果中没有的信息
4. 语气简洁专业；纯文本，可用「·」分点，不要 Markdown 代码块
5. 控制在 200 字以内，除非信息确实很多再略微放宽"""


def _truncate_json(obj: Any, max_chars: int = 3500) -> str:
    text = json.dumps(obj, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "…(已截断)"


def _extract_existing_answer(data: dict[str, Any]) -> str:
    """若业务侧已有可读文案，优先复用，避免重复调用 LLM。"""
    for key in ("answer", "summary", "readable_summary", "final_answer"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Master ReAct：最后一步 finish 的 final_answer
    trace = data.get("react_trace")
    if isinstance(trace, list) and trace:
        last = trace[-1]
        if isinstance(last, dict) and last.get("action") == "finish":
            fa = last.get("final_answer")
            if isinstance(fa, str) and fa.strip() and fa.strip() not in {"done", "任务完成"}:
                return fa.strip()
    return ""


def _heuristic_summary(
    *,
    success: bool,
    data: dict[str, Any],
    error_msg: str,
    reply_from: str,
) -> str:
    """无 API Key / 关闭整理时的兜底文案。"""
    existing = _extract_existing_answer(data)
    if existing:
        return existing

    status = "成功" if success else "失败"
    lines = [f"【{status}】来自 {reply_from or '系统'} 的任务结果。"]
    if error_msg:
        lines.append(f"· 错误：{error_msg}")

    sub = data.get("sub_results")
    if isinstance(sub, list) and sub:
        ok = sum(1 for s in sub if isinstance(s, dict) and s.get("success"))
        lines.append(f"· 子任务：{ok}/{len(sub)} 成功")
        for item in sub[:5]:
            if not isinstance(item, dict):
                continue
            agent = item.get("agent") or "?"
            if item.get("success"):
                lines.append(f"· {agent}：完成")
            else:
                lines.append(f"· {agent}：失败 — {item.get('error_msg') or '未知错误'}")

    sku = data.get("working_sku") or data.get("sku")
    if sku:
        lines.append(f"· 相关 SKU：{sku}")

    if len(lines) == 1:
        # 扁平业务字段：挑几个常见键
        highlights: list[str] = []
        for key in (
            "copy_draft",
            "suggest_stock_amount",
            "gross_profit",
            "profit_ratio",
            "is_risk",
            "risk_detail",
            "trans_text",
            "matched_sku",
            "best_sku",
        ):
            if key in data and data[key] not in (None, "", [], {}):
                highlights.append(f"· {key}：{data[key]}")
        if highlights:
            lines.extend(highlights[:6])
        else:
            lines.append("· 详见 data 字段中的结构化结果。")

    return "\n".join(lines)


async def polish_final_output(
    *,
    success: bool,
    data: dict[str, Any] | None,
    error_msg: str = "",
    user_query: str = "",
    reply_from: str = "",
    prefer_existing_answer: bool = True,
    on_provider_start: Callable[[], bool] | None = None,
    on_provider_result: Callable[[ChatResult], None] | None = None,
) -> str:
    """
    将最终结果整理为可读摘要。
    - 关闭开关 / 无 Key：启发式兜底
    - 已有 answer 等可读字段：默认直接返回（CRM 等已生成答复）
    - 否则调用当前 LLM Provider 整理
    """
    payload = data if isinstance(data, dict) else {}

    if not settings.OUTPUT_POLISH_ENABLED:
        return _heuristic_summary(
            success=success, data=payload, error_msg=error_msg, reply_from=reply_from
        )

    # 客服等已有自然语言答复：直接用，避免二次 LLM
    existing = _extract_existing_answer(payload)
    if prefer_existing_answer and isinstance(payload.get("answer"), str) and payload["answer"].strip():
        return payload["answer"].strip()

    if not is_llm_configured():
        return _heuristic_summary(
            success=success, data=payload, error_msg=error_msg, reply_from=reply_from
        )

    # 送入 LLM 的精简视图（去掉过大的嵌套）
    slim: dict[str, Any] = {
        "success": success,
        "error_msg": error_msg or None,
        "reply_from": reply_from or None,
        "user_query": (user_query or "")[:300] or None,
    }
    # Master：保留轨迹摘要与子结果要点
    if "sub_results" in payload:
        slim["sub_results"] = payload.get("sub_results")
        slim["working_sku"] = payload.get("working_sku")
        slim["all_success"] = payload.get("all_success")
        slim["timed_out"] = payload.get("timed_out")
        slim["plan"] = payload.get("plan")
        trace = payload.get("react_trace")
        if isinstance(trace, list):
            slim["react_trace_brief"] = [
                {
                    "step": t.get("step"),
                    "action": t.get("action"),
                    "thought": str(t.get("thought") or "")[:120],
                    "final_answer": t.get("final_answer"),
                }
                for t in trace
                if isinstance(t, dict)
            ]
    else:
        slim["data"] = payload

    if existing:
        slim["hint_final_answer"] = existing

    user_prompt = (
        "请整理以下任务结果：\n"
        f"{_truncate_json(slim, settings.OUTPUT_POLISH_MAX_INPUT_CHARS)}"
    )

    try:
        if on_provider_start is not None and not on_provider_start():
            return _heuristic_summary(
                success=success, data=payload, error_msg=error_msg, reply_from=reply_from
            )
        text = await llm_chat(
            user_prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=settings.OUTPUT_POLISH_MAX_TOKENS,
            mode="chat",
        )
        if on_provider_result is not None:
            on_provider_result(text)
        if text.content.strip():
            return text.content.strip()
    except Exception as exc:
        logger.warning(
            "output_polish_failed",
            extra={"event": "output_polish_failed", "error": str(exc)},
        )

    return _heuristic_summary(
        success=success, data=payload, error_msg=error_msg, reply_from=reply_from
    )
