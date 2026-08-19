"""拓展：库存预测工具。"""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill
from ecom_agent_matrix.db.base import AsyncPGClient


class StockPredictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sku: str = Field(min_length=1)
    predict_days: int = Field(default=7, ge=1, le=90)
    history_records: list[dict[str, Any]] = Field(default_factory=list)


class StockPredictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    daily_avg_sales: float = Field(ge=0)
    predict_cycle: int = Field(ge=1)
    suggest_stock_amount: int = Field(ge=0)
    base_suggest_stock_amount: int = Field(ge=0)
    history_used: int = Field(ge=0)
    history_adjusted: bool


@register_skill
class StockPredictTool(BaseSkill):
    read_only = True
    side_effect = False
    risk_level = "low"
    timeout_seconds = 15.0
    idempotent = True
    input_model = StockPredictInput
    output_model = StockPredictOutput
    skill_name = "stock_predict"
    skill_desc = (
        "商品库存备货预测，参数 sku、predict_days（默认7）、"
        "history_records（已废弃，仅保留调用兼容，不参与预测）"
    )

    async def run(self, params: dict) -> SkillResult:
        try:
            sku = params["sku"]
            predict_days = int(params.get("predict_days", 7))
            # history_records 可继续传入，但历史模型预测不是 observed truth，故完全忽略。

            # 统计近30天有效销量（剔除退款订单）
            stat_sql = """
            SELECT COALESCE(SUM(buy_num), 0)
            FROM ecom_order
            WHERE sku = %s AND create_time >= NOW() - INTERVAL '30 days' AND refund_flag = false;
            """
            stat_res = await AsyncPGClient.execute_sql(stat_sql, [sku])
            total_30d_sales = float(stat_res[0][0] or 0)
            daily_avg = total_30d_sales / 30
            safety_stock_rate = 1.2
            base_suggest = round(daily_avg * predict_days * safety_stock_rate)

            suggest_stock = base_suggest

            return SkillResult(
                success=True,
                data={
                    "daily_avg_sales": round(daily_avg, 2),
                    "predict_cycle": predict_days,
                    "suggest_stock_amount": suggest_stock,
                    "base_suggest_stock_amount": base_suggest,
                    "history_used": 0,
                    "history_adjusted": False,
                },
            )
        except KeyError as err:
            return SkillResult(success=False, error_msg=f"缺失参数：{err}")
        except ValueError:
            return SkillResult(success=False, error_msg="预测天数必须为整数")
