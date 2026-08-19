"""订单风险评估与风险记录原子 Skill。"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill
from ecom_agent_matrix.db.base import AsyncPGClient
from ecom_agent_matrix.core.security import tenant_scope_from_skill_context


class EvaluateOrderRiskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    order_no: str = Field(min_length=1)
    total_amount: float = Field(ge=0)
    buy_count: int = Field(ge=1)


class EvaluateOrderRiskOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_risk: bool
    risk_tags: list[str]
    risk_detail: str


class RecordOrderRiskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    order_no: str = Field(min_length=1)
    risk_type: str = Field(min_length=1)
    risk_desc: str = Field(min_length=1)


class RecordOrderRiskOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: int | None = None
    order_no: str
    risk_type: str
    risk_desc: str


def _evaluate(total_amount: float, buy_count: int) -> tuple[list[str], str]:
    tags: list[str] = []
    if total_amount > 500:
        tags.append("大额订单")
    if buy_count > 20:
        tags.append("批量囤货")
    return tags, "、".join(tags) if tags else "无异常风险"


async def _insert_risk(order_no: str, risk_type: str, risk_desc: str) -> int | None:
    scope = tenant_scope_from_skill_context()
    if not scope.usable:
        # Compatibility for legacy/dev direct calls. Trusted production execution
        # always takes the physical tenant/store path below.
        rows = await AsyncPGClient.execute_write(
            """
            INSERT INTO risk_record(order_no, risk_type, risk_desc)
            VALUES (%s, %s, %s) RETURNING id;
            """,
            [order_no, risk_type, risk_desc],
            scope=scope,
        )
        return rows[0][0] if rows and rows[0] else None
    rows = await AsyncPGClient.execute_write(
        """
        INSERT INTO risk_record(tenant_id, store_id, order_no, risk_type, risk_desc)
        VALUES (%s, %s, %s, %s, %s) RETURNING id;
        """,
        [scope.tenant_id, scope.store_id, order_no, risk_type, risk_desc],
        scope=scope,
    )
    return rows[0][0] if rows and rows[0] else None


@register_skill
class EvaluateOrderRiskTool(BaseSkill):
    read_only = True
    side_effect = False
    risk_level = "low"
    timeout_seconds = 5.0
    idempotent = True
    input_model = EvaluateOrderRiskInput
    output_model = EvaluateOrderRiskOutput
    skill_name = "evaluate_order_risk"
    skill_desc = "确定性订单风险评估，参数 order_no、total_amount、buy_count"

    async def run(self, params: dict) -> SkillResult:
        tags, detail = _evaluate(params["total_amount"], params["buy_count"])
        return SkillResult(
            success=True,
            data={"is_risk": bool(tags), "risk_tags": tags, "risk_detail": detail},
        )


@register_skill
class RecordOrderRiskTool(BaseSkill):
    read_only = False
    side_effect = True
    risk_level = "high"
    timeout_seconds = 10.0
    idempotent = False
    required_scopes = frozenset({"risk:write"})
    approval_required = True
    input_model = RecordOrderRiskInput
    output_model = RecordOrderRiskOutput
    skill_name = "record_order_risk"
    skill_desc = "写入已评估的订单风险，参数 order_no、risk_type、risk_desc"

    async def run(self, params: dict) -> SkillResult:
        record_id = await _insert_risk(
            params["order_no"], params["risk_type"], params["risk_desc"]
        )
        return SkillResult(success=True, data={"record_id": record_id, **params})


@register_skill
class OrderRiskControlTool(BaseSkill):
    """旧版组合入口；仅供现有调用兼容。"""

    read_only = False
    side_effect = True
    risk_level = "high"
    timeout_seconds = 15.0
    idempotent = False
    required_scopes = frozenset({"risk:write"})
    approval_required = True
    input_model = EvaluateOrderRiskInput
    output_model = EvaluateOrderRiskOutput
    deprecated = True
    replacement = "evaluate_order_risk + record_order_risk"
    skill_name = "order_risk_check"
    skill_desc = "[deprecated] 订单风险识别并记录；请改用原子风险 Skills"

    async def run(self, params: dict) -> SkillResult:
        tags, detail = _evaluate(params["total_amount"], params["buy_count"])
        if tags:
            await _insert_risk(params["order_no"], "order_abnormal", detail)
        return SkillResult(
            success=True,
            data={"is_risk": bool(tags), "risk_tags": tags, "risk_detail": detail},
        )
