"""竞品价格 Skill：只读监控计算与显式写入分离。"""
from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill
from ecom_agent_matrix.db.base import AsyncPGClient


class PriceMonitorInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )

    target_sku: str = Field(min_length=1)
    competitor: str = Field(min_length=1)
    compete_price: float = Field(gt=0)
    warn_threshold: float = Field(default=-10, le=0)


class PriceMonitorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    history_min_compete_price: float
    current_price_offset: float
    compete_price: float
    warn_threshold: float
    is_trigger_warn: bool
    warn_message: str


class RecordCompetitorPriceInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )

    target_sku: str = Field(min_length=1)
    competitor: str = Field(min_length=1)
    compete_price: float = Field(gt=0)


class RecordCompetitorPriceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    record_id: int | None
    target_sku: str
    competitor: str
    compete_price: float


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
    idempotent = True
    input_model = PriceMonitorInput
    output_model = PriceMonitorOutput
    skill_name = "price_monitor"
    skill_desc = (
        "只读查询竞品历史价格并判定告警，参数 target_sku、competitor、compete_price、"
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


@register_skill
class RecordCompetitorPrice(BaseSkill):
    """显式写入一条竞品价格记录，仅允许 Exec context 调用。"""

    read_only = False
    side_effect = True
    risk_level = "medium"
    idempotent = False
    input_model = RecordCompetitorPriceInput
    output_model = RecordCompetitorPriceOutput
    skill_name = "record_competitor_price"
    skill_desc = "记录新的竞品价格，参数 target_sku、competitor、compete_price"

    async def run(self, params: dict) -> SkillResult:
        try:
            target_sku = str(params["target_sku"]).strip()
            competitor = str(params["competitor"]).strip()
            compete_price = _as_float(params["compete_price"])
            if not target_sku or not competitor:
                return SkillResult(success=False, error_msg="target_sku / competitor 不能为空")

            insert_sql = """
            INSERT INTO competitor_price(target_sku, competitor_name, compete_price)
            VALUES (%s, %s, %s) RETURNING id;
            """
            rows = await AsyncPGClient.execute_sql(
                insert_sql,
                [target_sku, competitor, compete_price],
            )
            record_id = rows[0][0] if rows and rows[0] else None
            return SkillResult(
                success=True,
                data={
                    "record_id": record_id,
                    "target_sku": target_sku,
                    "competitor": competitor,
                    "compete_price": compete_price,
                },
            )
        except KeyError as exc:
            return SkillResult(success=False, error_msg=f"缺失参数：{exc}")
        except (TypeError, ValueError) as exc:
            return SkillResult(success=False, error_msg=f"竞品价格必须为数字：{exc}")
        except Exception as exc:
            return SkillResult(success=False, error_msg=f"竞品价格记录失败：{exc}")
