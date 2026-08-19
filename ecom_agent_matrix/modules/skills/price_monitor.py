"""竞品价格监控 Skill：入库、算偏移，并按阈值判定是否告警。"""
from __future__ import annotations

from decimal import Decimal

from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill
from ecom_agent_matrix.db.base import AsyncPGClient


def _as_float(value) -> float:
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _parse_warn_threshold(raw) -> tuple[float | None, str]:
    """下跌阈值必须 <= 0；返回 (值, 错误信息)。"""
    try:
        value = float(raw if raw is not None else -10)
    except (TypeError, ValueError):
        return None, "warn_threshold 必须为数字"
    if value > 0:
        return None, "warn_threshold 应为下跌阈值（<=0），例如 -10"
    return value, ""


@register_skill
class CompetitorPriceMonitor(BaseSkill):
    read_only = True
    side_effect = False
    risk_level = "medium"
    skill_name = "price_monitor"
    skill_desc = (
        "竞品价格写入并判定告警，参数 target_sku、competitor、compete_price、"
        "可选 warn_threshold（<=0，默认 -10）"
    )

    async def run(self, params: dict) -> SkillResult:
        try:
            target_sku = params["target_sku"]
            competitor = params["competitor"]
            compete_price = _as_float(params["compete_price"])
            warn_threshold, thr_err = _parse_warn_threshold(params.get("warn_threshold", -10))
            if thr_err:
                return SkillResult(success=False, error_msg=thr_err)

            insert_sql = """
            INSERT INTO competitor_price(target_sku, competitor_name, compete_price)
            VALUES (%s, %s, %s) RETURNING id;
            """
            insert_row = await AsyncPGClient.execute_sql(
                insert_sql, [target_sku, competitor, compete_price]
            )
            record_id = insert_row[0][0]

            min_sql = "SELECT MIN(compete_price) FROM competitor_price WHERE target_sku = %s;"
            min_price_row = await AsyncPGClient.execute_sql(min_sql, [target_sku])
            raw_min = min_price_row[0][0] if min_price_row and min_price_row[0] else None
            history_min = _as_float(raw_min) if raw_min is not None else compete_price
            price_diff = round(compete_price - history_min, 2)

            is_warn = price_diff <= float(warn_threshold)
            warn_msg = ""
            if is_warn:
                warn_msg = (
                    f"竞品 {competitor} 商品 {target_sku} 出现大幅降价，"
                    f"当前价 {compete_price}，相对历史最低偏移 {price_diff}"
                    f"（阈值 {warn_threshold}）"
                )

            return SkillResult(
                success=True,
                data={
                    "record_id": record_id,
                    "history_min_compete_price": history_min,
                    "current_price_offset": price_diff,
                    "compete_price": compete_price,
                    "warn_threshold": warn_threshold,
                    "is_trigger_warn": is_warn,
                    "warn_message": warn_msg,
                },
            )
        except KeyError as e:
            return SkillResult(success=False, error_msg=f"缺失参数：{e}")
        except (TypeError, ValueError) as e:
            return SkillResult(success=False, error_msg=f"竞品价格必须为数字：{e}")
        except Exception as e:
            return SkillResult(success=False, error_msg=f"竞品监控异常：{e}")
