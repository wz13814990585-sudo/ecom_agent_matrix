"""拓展：库存预测工具。"""
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill
from ecom_agent_matrix.db.base import AsyncPGClient


@register_skill
class StockPredictTool(BaseSkill):
    skill_name = "stock_predict"
    skill_desc = (
        "商品库存备货预测，参数 sku、predict_days（默认7）、"
        "history_records（可选，该 SKU 历史预测记忆）"
    )

    async def run(self, params: dict) -> SkillResult:
        try:
            sku = params["sku"]
            predict_days = int(params.get("predict_days", 7))
            history_records = params.get("history_records") or []

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

            # 结合同 SKU 历史建议量做平滑（若有历史记忆）
            hist_amounts = []
            for hit in history_records:
                meta = hit.get("meta") or {}
                if isinstance(meta, str):
                    continue
                amt = meta.get("suggest_stock_amount")
                if amt is not None:
                    try:
                        hist_amounts.append(float(amt))
                    except (TypeError, ValueError):
                        pass

            if hist_amounts:
                hist_avg = sum(hist_amounts) / len(hist_amounts)
                # 70% 当前销量模型 + 30% 历史建议，避免单次波动过大
                suggest_stock = round(base_suggest * 0.7 + hist_avg * 0.3)
                adjusted = True
            else:
                suggest_stock = base_suggest
                adjusted = False

            return SkillResult(
                success=True,
                data={
                    "daily_avg_sales": round(daily_avg, 2),
                    "predict_cycle": predict_days,
                    "suggest_stock_amount": suggest_stock,
                    "base_suggest_stock_amount": base_suggest,
                    "history_used": len(hist_amounts),
                    "history_adjusted": adjusted,
                },
            )
        except KeyError as err:
            return SkillResult(success=False, error_msg=f"缺失参数：{err}")
        except ValueError:
            return SkillResult(success=False, error_msg="预测天数必须为整数")
