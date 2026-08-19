"""运营报表聚合 Skill（SQL 统计 + 结构化 LLM 摘要）。"""
from __future__ import annotations

import json
import re

from ecom_agent_matrix.config.constants import (
    TABLE_COMPETITOR,
    TABLE_GOODS,
    TABLE_ORDER,
    TABLE_RISK_LOG,
)
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm.deepseek_client import deepseek_chat
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill
from ecom_agent_matrix.db.base import AsyncPGClient

SUPPORTED_REPORT_TYPES = frozenset({"daily_ops", "sales", "stock", "risk", "full"})


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if block:
        text = block.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    return json.loads(text)


async def _sales_stats(days: int) -> dict:
    sql = f"""
    SELECT
      COUNT(*) AS order_cnt,
      COALESCE(SUM(buy_num), 0) AS units,
      COALESCE(SUM(total_amount), 0) AS gmv,
      COALESCE(SUM(CASE WHEN refund_flag THEN 1 ELSE 0 END), 0) AS refund_orders
    FROM {TABLE_ORDER}
    WHERE create_time >= NOW() - make_interval(days => %s)
    """
    rows = await AsyncPGClient.execute_sql(sql, [days])
    r = rows[0] if rows else (0, 0, 0, 0)
    order_cnt, units, gmv, refund_orders = r
    gmv_f = float(gmv or 0)
    return {
        "days": days,
        "order_count": int(order_cnt or 0),
        "units_sold": int(units or 0),
        "gmv": round(gmv_f, 2),
        "refund_orders": int(refund_orders or 0),
        "refund_rate": round(int(refund_orders or 0) / max(int(order_cnt or 0), 1), 4),
    }


async def _top_skus(days: int, top_k: int = 5) -> list[dict]:
    sql = f"""
    SELECT o.sku, COALESCE(SUM(o.buy_num), 0) AS units, COALESCE(SUM(o.total_amount), 0) AS gmv
    FROM {TABLE_ORDER} o
    WHERE o.create_time >= NOW() - make_interval(days => %s)
      AND COALESCE(o.refund_flag, false) = false
    GROUP BY o.sku
    ORDER BY units DESC
    LIMIT %s
    """
    rows = await AsyncPGClient.execute_sql(sql, [days, top_k])
    return [
        {"sku": r[0], "units": int(r[1] or 0), "gmv": round(float(r[2] or 0), 2)}
        for r in rows
    ]


async def _stock_stats() -> dict:
    sql = f"""
    SELECT
      COUNT(*) AS sku_cnt,
      COALESCE(SUM(stock_num), 0) AS total_stock,
      COALESCE(SUM(CASE WHEN stock_num <= 0 THEN 1 ELSE 0 END), 0) AS oos_cnt,
      COALESCE(SUM(CASE WHEN stock_num > 0 AND stock_num < 20 THEN 1 ELSE 0 END), 0) AS low_stock_cnt
    FROM {TABLE_GOODS}
    """
    rows = await AsyncPGClient.execute_sql(sql, [])
    r = rows[0] if rows else (0, 0, 0, 0)
    return {
        "sku_count": int(r[0] or 0),
        "total_stock": int(r[1] or 0),
        "out_of_stock_skus": int(r[2] or 0),
        "low_stock_skus": int(r[3] or 0),
    }


async def _risk_stats(days: int) -> dict:
    sql = f"""
    SELECT risk_type, COUNT(*) AS cnt
    FROM {TABLE_RISK_LOG}
    WHERE create_time >= NOW() - make_interval(days => %s)
    GROUP BY risk_type
    ORDER BY cnt DESC
    """
    rows = await AsyncPGClient.execute_sql(sql, [days])
    by_type = {str(r[0]): int(r[1] or 0) for r in rows}
    return {"days": days, "total": sum(by_type.values()), "by_type": by_type}


async def _competitor_stats(days: int) -> dict:
    sql = f"""
    SELECT COUNT(*) AS records, COUNT(DISTINCT target_sku) AS skus, COUNT(DISTINCT competitor_name) AS shops
    FROM {TABLE_COMPETITOR}
    WHERE crawl_time >= NOW() - make_interval(days => %s)
    """
    rows = await AsyncPGClient.execute_sql(sql, [days])
    r = rows[0] if rows else (0, 0, 0)
    return {
        "days": days,
        "price_records": int(r[0] or 0),
        "monitored_skus": int(r[1] or 0),
        "competitors": int(r[2] or 0),
    }


def _template_summary(report_type: str, sections: dict) -> str:
    lines = [f"【{report_type} 运营简报】"]
    anomalies: list[str] = []
    if "sales" in sections:
        s = sections["sales"]
        lines.append(
            f"近{s['days']}天：订单 {s['order_count']}，销量 {s['units_sold']}，"
            f"GMV {s['gmv']}，退款率 {s['refund_rate']*100:.1f}%"
        )
        if s.get("refund_rate", 0) >= 0.08:
            anomalies.append(f"退款率偏高 {s['refund_rate']*100:.1f}%")
    if "top_skus" in sections and sections["top_skus"]:
        top = ", ".join(f"{x['sku']}({x['units']})" for x in sections["top_skus"][:3])
        lines.append(f"热销 SKU：{top}")
    if "stock" in sections:
        st = sections["stock"]
        lines.append(
            f"库存：SKU {st['sku_count']}，总库存 {st['total_stock']}，"
            f"缺货 {st['out_of_stock_skus']}，低库存 {st['low_stock_skus']}"
        )
        if st.get("out_of_stock_skus", 0) > 0:
            anomalies.append(f"缺货 SKU {st['out_of_stock_skus']} 个")
        if st.get("low_stock_skus", 0) > 0:
            anomalies.append(f"低库存 SKU {st['low_stock_skus']} 个")
    if "risk" in sections:
        rk = sections["risk"]
        lines.append(f"风控：近{rk['days']}天记录 {rk['total']} 条")
        if rk.get("total", 0) > 0:
            anomalies.append(f"近{rk['days']}天风控 {rk['total']} 条")
    if "competitor" in sections:
        c = sections["competitor"]
        lines.append(
            f"竞品：近{c['days']}天报价 {c['price_records']} 条，覆盖 SKU {c['monitored_skus']}"
        )
    if anomalies:
        lines.append("异常点：" + "；".join(anomalies))
        lines.append("建议动作：1) 核对异常指标明细 2) 优先处理缺货/高退款 SKU 3) 复盘投放与定价")
    else:
        lines.append("异常点：未见明显异常")
        lines.append("建议动作：1) 维持日常监控 2) 关注热销备货 3) 下周复盘 GMV 与退款")
    return "\n".join(lines)


def _format_structured_summary(parsed: dict, fallback: str) -> str:
    anomalies = parsed.get("anomalies") if isinstance(parsed.get("anomalies"), list) else []
    hypotheses = parsed.get("hypotheses") if isinstance(parsed.get("hypotheses"), list) else []
    actions = parsed.get("actions") if isinstance(parsed.get("actions"), list) else []
    brief = str(parsed.get("summary") or "").strip()
    if not (anomalies or hypotheses or actions or brief):
        return fallback
    lines: list[str] = []
    if brief:
        lines.append(brief)
    if anomalies:
        lines.append("异常点：")
        lines.extend(f"- {str(x).strip()}" for x in anomalies[:5] if str(x).strip())
    if hypotheses:
        lines.append("可能原因：")
        lines.extend(f"- {str(x).strip()}" for x in hypotheses[:4] if str(x).strip())
    if actions:
        lines.append("建议动作：")
        for i, act in enumerate(actions[:3], 1):
            text = str(act).strip()
            if text:
                lines.append(f"{i}) {text}")
    return "\n".join(lines) if lines else fallback


@register_skill
class OpsReportTool(BaseSkill):
    skill_name = "ops_report"
    skill_desc = (
        "运营报表聚合，参数 report_type=daily_ops|sales|stock|risk|full、"
        "days 统计天数、top_k 热销条数、lang"
    )

    async def run(self, params: dict) -> SkillResult:
        try:
            report_type = str(params.get("report_type") or "daily_ops").strip().lower()
            if report_type not in SUPPORTED_REPORT_TYPES:
                return SkillResult(
                    success=False,
                    error_msg=(
                        f"不支持的 report_type：{report_type}，"
                        f"可选：{', '.join(sorted(SUPPORTED_REPORT_TYPES))}"
                    ),
                )
            days = int(params.get("days", 7))
            top_k = int(params.get("top_k", 5))
            lang = str(params.get("lang") or "zh").strip().lower()
            if days <= 0 or days > 90:
                return SkillResult(success=False, error_msg="days 需在 1~90")
            if top_k <= 0 or top_k > 20:
                return SkillResult(success=False, error_msg="top_k 需在 1~20")

            sections: dict = {}
            need_sales = report_type in ("daily_ops", "sales", "full")
            need_stock = report_type in ("daily_ops", "stock", "full")
            need_risk = report_type in ("daily_ops", "risk", "full")
            need_comp = report_type in ("daily_ops", "full")

            if need_sales:
                sections["sales"] = await _sales_stats(days)
                sections["top_skus"] = await _top_skus(days, top_k)
            if need_stock:
                sections["stock"] = await _stock_stats()
            if need_risk:
                sections["risk"] = await _risk_stats(days)
            if need_comp:
                sections["competitor"] = await _competitor_stats(days)

            source = "template"
            llm_error = ""
            summary = _template_summary(report_type, sections)
            structured: dict = {}

            use_llm = bool(settings.DEEPSEEK_API_KEY) and bool(
                getattr(settings, "AGENT_LLM_EXPLAIN_ENABLED", True)
            )
            if use_llm:
                try:
                    user_prompt = (
                        f"Language: {lang}\n"
                        f"Report type: {report_type}\n"
                        f"Metrics JSON:\n{json.dumps(sections, ensure_ascii=False)}\n\n"
                        "Return ONLY JSON with keys:\n"
                        '- "summary": one short paragraph\n'
                        '- "anomalies": string array (key issues; empty if none)\n'
                        '- "hypotheses": string array (likely causes)\n'
                        '- "actions": exactly 3 actionable next steps\n'
                        "Do not invent metrics not present in the JSON."
                    )
                    text = (
                        await deepseek_chat(
                            user_prompt=user_prompt,
                            system_prompt=(
                                "You are an ecommerce operations analyst for a cross-border store. "
                                "Be factual, actionable, and brief. Output JSON only."
                            ),
                            temperature=0.2,
                            max_tokens=int(
                                getattr(settings, "AGENT_LLM_EXPLAIN_MAX_TOKENS", 450) or 450
                            ),
                            mode="chat",
                        )
                    ).content.strip()
                    if text:
                        structured = _extract_json(text)
                        summary = _format_structured_summary(structured, summary)
                        source = "deepseek"
                    else:
                        llm_error = "empty_content"
                except Exception as exc:
                    llm_error = str(exc)

            return SkillResult(
                success=True,
                data={
                    "report_type": report_type,
                    "days": days,
                    "sections": sections,
                    "summary": summary.strip(),
                    "structured": {
                        "anomalies": structured.get("anomalies") or [],
                        "hypotheses": structured.get("hypotheses") or [],
                        "actions": structured.get("actions") or [],
                    }
                    if structured
                    else {},
                    "source": source,
                    "lang": lang,
                    "llm_error": llm_error,
                },
            )
        except ValueError:
            return SkillResult(success=False, error_msg="days/top_k 必须为整数")
        except Exception as exc:
            return SkillResult(success=False, error_msg=f"报表生成失败：{exc}")
