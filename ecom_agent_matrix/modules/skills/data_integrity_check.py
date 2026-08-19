"""业务数据完整性校验 Skill。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ecom_agent_matrix.config.constants import TABLE_GOODS, TABLE_ORDER
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill
from ecom_agent_matrix.db.base import AsyncPGClient

SUPPORTED_SCOPES = frozenset({"goods", "order", "full"})


class DataIntegrityCheckInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    scope: Literal["goods", "order", "full"] = "full"
    sku: str | None = None
    target_sku: str | None = None
    order_no: str | None = None
    limit: int = Field(default=50, ge=1, le=500)


class DataIntegrityCheckOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: str
    sku: str | None
    order_no: str | None
    issue_count: int = Field(ge=0)
    passed: bool
    issue_types: dict[str, int]
    issues: list[dict[str, Any]]


async def _check_goods(sku: str | None = None, limit: int = 50) -> list[dict]:
    issues: list[dict] = []
    params: list = []
    where = "WHERE 1=1"
    if sku:
        where += " AND sku = %s"
        params.append(sku)

    sql = f"""
    SELECT sku, price, stock_num, title_zh, title_en, category
    FROM {TABLE_GOODS}
    {where}
    ORDER BY id DESC
    LIMIT %s
    """
    params.append(limit)
    rows = await AsyncPGClient.execute_sql(sql, params)
    for r in rows:
        g_sku, price, stock, title_zh, title_en, category = r
        problems = []
        if price is None or float(price) <= 0:
            problems.append("price_invalid")
        if stock is None or int(stock) < 0:
            problems.append("stock_negative")
        if not (title_zh or title_en):
            problems.append("title_missing")
        if not category:
            problems.append("category_missing")
        if problems:
            issues.append(
                {
                    "entity": "goods",
                    "sku": g_sku,
                    "problems": problems,
                    "price": float(price) if price is not None else None,
                    "stock_num": stock,
                }
            )
    return issues


async def _check_orders(order_no: str | None = None, sku: str | None = None, limit: int = 50) -> list[dict]:
    issues: list[dict] = []
    params: list = []
    where = "WHERE 1=1"
    if order_no:
        where += " AND o.order_no = %s"
        params.append(order_no)
    if sku:
        where += " AND o.sku = %s"
        params.append(sku)

    sql = f"""
    SELECT o.order_no, o.sku, o.buy_num, o.total_amount, o.refund_flag, g.sku AS goods_sku, g.price
    FROM {TABLE_ORDER} o
    LEFT JOIN {TABLE_GOODS} g ON g.sku = o.sku
    {where}
    ORDER BY o.id DESC
    LIMIT %s
    """
    params.append(limit)
    rows = await AsyncPGClient.execute_sql(sql, params)
    for r in rows:
        ono, o_sku, buy_num, total_amount, refund_flag, goods_sku, goods_price = r
        problems = []
        if not o_sku:
            problems.append("sku_empty")
        if goods_sku is None:
            problems.append("orphan_sku")
        if buy_num is None or int(buy_num) <= 0:
            problems.append("buy_num_invalid")
        if total_amount is None or float(total_amount) < 0:
            problems.append("amount_invalid")
        if (
            goods_price is not None
            and buy_num is not None
            and total_amount is not None
            and int(buy_num) > 0
        ):
            expect = float(goods_price) * int(buy_num)
            # 允许小额误差（折扣/运费场景下仅作提示）
            if abs(float(total_amount) - expect) > max(1.0, expect * 0.35):
                problems.append("amount_mismatch")
        if problems:
            issues.append(
                {
                    "entity": "order",
                    "order_no": ono,
                    "sku": o_sku,
                    "problems": problems,
                    "buy_num": buy_num,
                    "total_amount": float(total_amount) if total_amount is not None else None,
                    "refund_flag": bool(refund_flag),
                    "goods_price": float(goods_price) if goods_price is not None else None,
                }
            )
    return issues


@register_skill
class DataIntegrityCheckTool(BaseSkill):
    read_only = True
    side_effect = False
    risk_level = "medium"
    timeout_seconds = 15.0
    idempotent = True
    input_model = DataIntegrityCheckInput
    output_model = DataIntegrityCheckOutput
    skill_name = "data_integrity_check"
    skill_desc = (
        "电商数据完整性校验，参数 scope=goods|order|full、"
        "可选 sku / order_no / limit"
    )

    async def run(self, params: dict) -> SkillResult:
        try:
            scope = str(params.get("scope") or "full").strip().lower()
            if scope not in SUPPORTED_SCOPES:
                return SkillResult(
                    success=False,
                    error_msg=f"不支持的 scope：{scope}，可选：{', '.join(sorted(SUPPORTED_SCOPES))}",
                )
            sku = str(params.get("sku") or params.get("target_sku") or "").strip() or None
            order_no = str(params.get("order_no") or "").strip() or None
            limit = int(params.get("limit", 50))
            if limit <= 0 or limit > 500:
                return SkillResult(success=False, error_msg="limit 需在 1~500")

            issues: list[dict] = []
            if scope in ("goods", "full"):
                issues.extend(await _check_goods(sku=sku, limit=limit))
            if scope in ("order", "full"):
                issues.extend(await _check_orders(order_no=order_no, sku=sku, limit=limit))

            by_type: dict[str, int] = {}
            for item in issues:
                for p in item.get("problems") or []:
                    by_type[p] = by_type.get(p, 0) + 1

            return SkillResult(
                success=True,
                data={
                    "scope": scope,
                    "sku": sku,
                    "order_no": order_no,
                    "issue_count": len(issues),
                    "passed": len(issues) == 0,
                    "issue_types": by_type,
                    "issues": issues[:100],
                },
            )
        except ValueError:
            return SkillResult(success=False, error_msg="limit 必须为整数")
        except Exception as exc:
            return SkillResult(success=False, error_msg=f"数据校验异常：{type(exc).__name__}")
