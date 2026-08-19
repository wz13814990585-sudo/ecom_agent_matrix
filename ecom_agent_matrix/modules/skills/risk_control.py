"""拓展：订单风控工具。"""
# modules/skills/risk_control.py
from ecom_agent_matrix.db.base import AsyncPGClient
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill

@register_skill
class OrderRiskControlTool(BaseSkill):
    read_only = False
    side_effect = True
    risk_level = "high"
    skill_name = "order_risk_check"
    skill_desc = "订单风险识别，参数order_no订单号、total_amount订单总金额、buy_count下单数量"

    async def run(self, params: dict) -> SkillResult:
        try:
            order_no = params["order_no"]
            total_amount = float(params["total_amount"])
            buy_count = int(params["buy_count"])
            risk_tags = []

            # 风控规则
            if total_amount > 500:
                risk_tags.append("大额订单")
            if buy_count > 20:
                risk_tags.append("批量囤货")

            if risk_tags:
                risk_desc = "、".join(risk_tags)
                insert_sql = """
                INSERT INTO risk_record(order_no, risk_type, risk_desc)
                VALUES (%s, 'order_abnormal', %s);
                """
                await AsyncPGClient.execute_sql(insert_sql, [order_no, risk_desc])
                return SkillResult(success=True, data={"is_risk": True, "risk_detail": risk_desc})
            else:
                return SkillResult(success=True, data={"is_risk": False, "risk_detail": "无异常风险"})
        except KeyError as e:
            return SkillResult(success=False, error_msg=f"缺失参数：{e}")
