"""利润/库存测算工具。"""
# modules/skills/calc_tool.py
from pydantic import BaseModel, ConfigDict, Field

from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill


class ProfitCalcInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    cost: float = Field(ge=0)
    shipping: float = Field(ge=0)
    commission_rate: float = Field(ge=0, lt=1)
    sell_price: float = Field(ge=0)


class ProfitCalcOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    unit_total_cost: float
    gross_profit: float
    profit_ratio: float
    break_even_price: float


@register_skill
class ProfitCalcTool(BaseSkill):
    read_only = True
    side_effect = False
    risk_level = "low"
    idempotent = True
    input_model = ProfitCalcInput
    output_model = ProfitCalcOutput
    skill_name = "profit_calc"
    skill_desc = "跨境独立站利润测算，参数：cost采购价、shipping物流分摊、commission_rate平台佣金、sell_price售价"

    async def run(self, params: dict) -> SkillResult:
        try:
            cost = float(params["cost"])
            shipping = float(params["shipping"])
            commission_rate = float(params["commission_rate"])
            sell_price = float(params["sell_price"])

            total_fixed_cost = cost + shipping
            commission_fee = sell_price * commission_rate
            total_cost = total_fixed_cost + commission_fee

            gross_profit = sell_price - total_cost
            profit_ratio = gross_profit / sell_price if sell_price != 0 else 0
            break_even_price = total_fixed_cost / (1 - commission_rate)

            return SkillResult(
                success=True,
                data={
                    "unit_total_cost": round(total_cost, 2),
                    "gross_profit": round(gross_profit, 2),
                    "profit_ratio": round(profit_ratio, 3),
                    "break_even_price": round(break_even_price, 2)
                }
            )
        except KeyError as e:
            return SkillResult(success=False, error_msg=f"缺失必填参数：{str(e)}")
        except ValueError:
            return SkillResult(success=False, error_msg="成本、售价、佣金必须为数字")
