"""广告投放优化 Skill（DeepSeek + 规则兜底）。"""
from __future__ import annotations

import json
import re

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm.deepseek_client import deepseek_chat
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill

SUPPORTED_AD_PLATFORMS = frozenset({"meta", "google", "tiktok", "amazon"})

AD_SYSTEM_PROMPT = """You are a cross-border ecommerce paid-ads optimizer.
Given campaign metrics, return ONLY valid JSON (no markdown):
{
  "action": "scale_up|scale_down|pause|hold|restructure",
  "bid_adjust_pct": -20,
  "budget_adjust_pct": 10,
  "target_roas": 2.5,
  "priority": "high|medium|low",
  "reasoning": "one short sentence",
  "checklist": ["action1", "action2"]
}
Rules:
- bid_adjust_pct / budget_adjust_pct are integers -50..50
- Be conservative when data is sparse (low clicks/conversions)
"""


def _safe_float(val, default: float | None = None) -> float | None:
    if val is None or str(val).strip() == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _rule_optimize(metrics: dict) -> dict:
    """无 LLM 时的规则优化。"""
    spend = float(metrics.get("spend") or 0)
    clicks = float(metrics.get("clicks") or 0)
    conversions = float(metrics.get("conversions") or 0)
    revenue = float(metrics.get("revenue") or 0)
    roas = (revenue / spend) if spend > 0 else 0.0
    cpc = (spend / clicks) if clicks > 0 else 0.0
    cpa = (spend / conversions) if conversions > 0 else 0.0
    target_roas = float(metrics.get("target_roas") or 2.0)

    if spend <= 0 and clicks <= 0:
        action, bid_pct, budget_pct, priority = "hold", 0, 0, "low"
        reasoning = "缺少投放数据，建议先小预算冷启动收集转化"
        checklist = ["设置转化像素", "日预算试投", "观察 3 天 CTR/CPC"]
    elif conversions == 0 and spend >= max(30.0, target_roas * 10):
        action, bid_pct, budget_pct, priority = "scale_down", -15, -30, "high"
        reasoning = "有消耗无转化，先降预算并检查落地页/素材"
        checklist = ["暂停高 CPC 广告组", "检查落地页加载", "替换主图/标题"]
    elif roas >= target_roas * 1.2:
        action, bid_pct, budget_pct, priority = "scale_up", 10, 25, "medium"
        reasoning = f"ROAS={roas:.2f} 高于目标 {target_roas}，建议放量"
        checklist = ["提高日预算 20%~30%", "复制赢家广告组", "扩展相似受众"]
    elif roas >= target_roas * 0.8:
        action, bid_pct, budget_pct, priority = "hold", 0, 0, "low"
        reasoning = f"ROAS={roas:.2f} 接近目标，维持观察"
        checklist = ["保持预算", "A/B 测试创意", "关注频次疲劳"]
    else:
        action, bid_pct, budget_pct, priority = "scale_down", -10, -20, "medium"
        reasoning = f"ROAS={roas:.2f} 低于目标 {target_roas}，收紧投放"
        checklist = ["降低出价", "关掉低转化词/受众", "优化商品页"]

    return {
        "action": action,
        "bid_adjust_pct": bid_pct,
        "budget_adjust_pct": budget_pct,
        "target_roas": target_roas,
        "priority": priority,
        "reasoning": reasoning,
        "checklist": checklist,
        "metrics_snapshot": {
            "spend": round(spend, 2),
            "clicks": int(clicks),
            "conversions": int(conversions),
            "revenue": round(revenue, 2),
            "roas": round(roas, 3),
            "cpc": round(cpc, 3),
            "cpa": round(cpa, 3) if conversions else None,
        },
    }


def _extract_json(text: str) -> dict:
    text = text.strip()
    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if block:
        text = block.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    return json.loads(text)


def _clamp_pct(val, default: int = 0) -> int:
    try:
        n = int(round(float(val)))
    except (TypeError, ValueError):
        return default
    return max(-50, min(50, n))


@register_skill
class AdOptimizeTool(BaseSkill):
    skill_name = "ad_optimize"
    skill_desc = (
        "广告投放优化建议，参数 sku/campaign_id、platform(meta/google/tiktok/amazon)、"
        "spend/clicks/conversions/revenue、target_roas、daily_budget、bid"
    )

    async def run(self, params: dict) -> SkillResult:
        try:
            platform = str(params.get("platform") or "meta").strip().lower()
            if platform not in SUPPORTED_AD_PLATFORMS:
                return SkillResult(
                    success=False,
                    error_msg=(
                        f"不支持的广告平台：{platform}，"
                        f"可选：{', '.join(sorted(SUPPORTED_AD_PLATFORMS))}"
                    ),
                )

            metrics = {
                "sku": str(params.get("sku") or params.get("target_sku") or "").strip(),
                "campaign_id": str(params.get("campaign_id") or "").strip(),
                "platform": platform,
                "spend": _safe_float(params.get("spend"), 0.0) or 0.0,
                "clicks": _safe_float(params.get("clicks"), 0.0) or 0.0,
                "conversions": _safe_float(params.get("conversions"), 0.0) or 0.0,
                "revenue": _safe_float(params.get("revenue"), 0.0) or 0.0,
                "daily_budget": _safe_float(params.get("daily_budget")),
                "bid": _safe_float(params.get("bid")),
                "target_roas": _safe_float(params.get("target_roas"), 2.0) or 2.0,
            }

            source = "rules"
            llm_error = ""
            plan = _rule_optimize(metrics)

            if settings.DEEPSEEK_API_KEY:
                try:
                    user_prompt = (
                        "Optimize this ecommerce ad campaign:\n"
                        f"{json.dumps(metrics, ensure_ascii=False)}\n"
                        f"Rule baseline (may revise):\n{json.dumps(plan, ensure_ascii=False)}"
                    )
                    raw = await deepseek_chat(
                        user_prompt=user_prompt,
                        system_prompt=AD_SYSTEM_PROMPT,
                        temperature=0.2,
                        max_tokens=600,
                        mode="chat",
                    )
                    parsed = _extract_json(raw.content)
                    plan = {
                        "action": str(parsed.get("action") or plan["action"]),
                        "bid_adjust_pct": _clamp_pct(
                            parsed.get("bid_adjust_pct"), plan["bid_adjust_pct"]
                        ),
                        "budget_adjust_pct": _clamp_pct(
                            parsed.get("budget_adjust_pct"), plan["budget_adjust_pct"]
                        ),
                        "target_roas": float(
                            parsed.get("target_roas") or plan["target_roas"]
                        ),
                        "priority": str(parsed.get("priority") or plan["priority"]),
                        "reasoning": str(parsed.get("reasoning") or plan["reasoning"]),
                        "checklist": (
                            parsed.get("checklist")
                            if isinstance(parsed.get("checklist"), list)
                            else plan["checklist"]
                        ),
                        "metrics_snapshot": plan["metrics_snapshot"],
                    }
                    source = "deepseek"
                except Exception as exc:
                    llm_error = str(exc)

            # 应用建议到预算/出价（若有输入）
            suggested = {}
            if metrics.get("daily_budget") is not None:
                suggested["daily_budget"] = round(
                    metrics["daily_budget"] * (1 + plan["budget_adjust_pct"] / 100), 2
                )
            if metrics.get("bid") is not None:
                suggested["bid"] = round(
                    metrics["bid"] * (1 + plan["bid_adjust_pct"] / 100), 2
                )

            return SkillResult(
                success=True,
                data={
                    "sku": metrics["sku"],
                    "campaign_id": metrics["campaign_id"],
                    "platform": platform,
                    "plan": plan,
                    "suggested": suggested,
                    "source": source,
                    "llm_error": llm_error,
                },
            )
        except Exception as exc:
            return SkillResult(success=False, error_msg=f"广告优化失败：{exc}")
