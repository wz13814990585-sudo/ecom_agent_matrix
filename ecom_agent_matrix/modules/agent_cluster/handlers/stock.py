"""库存查询 handler：只读预测 / 备货建议。由 Query Agent 调用，不是独立 Agent。"""
import time

from ecom_agent_matrix.config.constants import AGENT_QUERY
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.memory.long_vector_memory import AgentLongVectorMemory
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.modules.utils.competitor_parse import extract_sku
from ecom_agent_matrix.modules.utils.llm_explain import llm_explain

logger = setup_logger("agent.stock")

_long_mem: AgentLongVectorMemory | None = None


def _mem() -> AgentLongVectorMemory:
    global _long_mem
    if _long_mem is None:
        _long_mem = AgentLongVectorMemory()
    return _long_mem


async def handle_stock(payload: dict) -> tuple[bool, str, dict]:
    """按 SKU 召回历史预测 → stock_predict → 可选写记忆。"""
    started = time.perf_counter()
    sku = extract_sku(payload)
    predict_days = int(payload.get("predict_days", 7))
    history_hits: list = []
    long_mem = _mem()

    if not sku:
        return (
            False,
            "缺少 sku，请先解析商品名或直接提供 SKU",
            {"query_kind": "stock", "sku": "", "predict_days": predict_days, "history_hits": 0},
        )

    if predict_days <= 0:
        return (
            False,
            "predict_days 必须为正整数",
            {"query_kind": "stock", "sku": sku, "predict_days": predict_days},
        )

    try:
        history_hits = await long_mem.recall(
            query_text=f"sku:{sku} 库存预测 备货",
            agent_name=AGENT_QUERY,
            top_k=5,
            meta_filter={"sku": sku},
        )
    except Exception as exc:
        logger.warning(
            "stock_memory_recall_failed",
            extra={"event": "stock_memory_recall_failed", "error": str(exc)},
        )
        history_hits = []

    skill_result = await exec_skill(
        "stock_predict",
        {
            "sku": sku,
            "predict_days": predict_days,
            "history_records": history_hits,
        },
    )

    suggest = (skill_result.data or {}).get("suggest_stock_amount")
    if skill_result.success:
        await long_mem.safe_save_memory(
            agent_name=AGENT_QUERY,
            content=(
                f"sku:{sku},预测周期{predict_days}天,"
                f"建议库存:{suggest},"
                f"历史召回:{len(history_hits)}条"
            ),
            meta={
                "sku": sku,
                "predict_days": predict_days,
                "suggest_stock_amount": suggest,
                "history_hits": len(history_hits),
                "success": True,
                "confidence": 0.8,
                "deprecated": False,
            },
        )

    advice = ""
    advice_source = ""
    advice_error = ""
    if skill_result.success:
        pred = skill_result.data or {}
        daily = pred.get("daily_avg_sales")
        advice_fallback = (
            f"SKU {sku} 近30日日均销量约 {daily}，"
            f"{predict_days} 天建议备货量 {suggest}（含安全库存系数）。"
            "请结合在途库存与促销计划调整。"
        )
        advice, advice_source, advice_error = await llm_explain(
            system_prompt=(
                "你是跨境电商补货顾问。根据预测数字写简短中文说明："
                "风险点与补货建议。不要改写数字，不要编造未提供的供应商交期。"
            ),
            user_prompt=(
                f"sku={sku}\npredict_days={predict_days}\n"
                f"prediction={pred}\nhistory_hits={len(history_hits)}"
            ),
            fallback=advice_fallback,
            max_tokens=int(getattr(settings, "AGENT_LLM_EXPLAIN_MAX_TOKENS", 450) or 450),
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    return (
        skill_result.success,
        skill_result.error_msg if not skill_result.success else "",
        {
            "query_kind": "stock",
            "sku": sku,
            "predict_days": predict_days,
            "history_hits": len(history_hits),
            "history_preview": [
                {
                    "id": h.get("id"),
                    "content": h.get("content"),
                    "meta": h.get("meta"),
                    "distance": h.get("distance"),
                }
                for h in history_hits[:3]
            ],
            "stock_predict_result": skill_result.data,
            "advice": advice,
            "advice_source": advice_source,
            "advice_error": advice_error or None,
            "latency_ms": round(elapsed_ms, 2),
        },
    )
