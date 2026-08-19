"""请求/响应模型（含 OpenAPI 示例）。"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

RESERVED_SECURITY_FIELDS = frozenset(
    {
        "tenant_id", "user_id", "store_id", "roles", "role", "scopes", "scope",
        "subject", "_security", "security_context", "auth_context",
    }
)


class TaskCreateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"query": "查询数据库有哪些商品"},
                {"query": "为「防水户外背包」生成tiktok文案"},
                {
                    "query": "SKU-BAG-001 需要备货多少天",
                    "task_type": "stock_analysis",
                    "payload": {"sku": "SKU-BAG-001", "predict_days": 14},
                },
            ]
        }
    )

    query: str = Field(..., min_length=1, description="自然语言任务（只填这一项即可）")
    task_type: Optional[str] = Field(
        default=None,
        description=(
            "可选；不填则自动识别。可选值：knowledge_qa / stock_analysis / "
            "social_marketing / competitor_watch / goods_search / goods_catalog / "
            "ad_optimize / ad_query / data_check / order_query / ops_report / risk_control"
        ),
    )
    priority: Optional[int] = Field(
        default=None, ge=0, le=4, description="可选；0最高，4最低。Swagger 请留空不要填 0 以外的空串"
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="可选额外字段；日常可删掉整段",
    )
    timeout: Optional[float] = Field(
        default=None,
        description="可选等待秒数；Swagger 请删除该字段或留 null，不要填空字符串/0",
    )

    @field_validator("task_type", mode="before")
    @classmethod
    def _empty_str_task_type(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v

    @field_validator("priority", "timeout", mode="before")
    @classmethod
    def _empty_optional_number(cls, v):
        # Swagger UI 常把未填项发成 ""，或把 timeout 填成 0 → 直接当未设置
        if v is None or v == "" or v == 0 or v == 0.0:
            return None
        return v

    @field_validator("payload")
    @classmethod
    def _forbid_spoofed_identity(cls, value: dict[str, Any]) -> dict[str, Any]:
        forbidden = RESERVED_SECURITY_FIELDS.intersection(value)
        if forbidden:
            raise ValueError("SECURITY_FIELD_FORBIDDEN: " + ", ".join(sorted(forbidden)))
        return value


class CustomerChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "这款背包材质是什么，怎么清洗？",
                    "lang": "zh",
                    "use_rag": True,
                    "session_id": "sess-demo-001",
                },
                {
                    "query": "帮我查一下订单 ORD-20260816-001",
                    "lang": "zh",
                    "use_taobao": True,
                    "order_no": "123456789012345678",
                },
            ]
        }
    )

    query: str = Field(..., min_length=1)
    session_id: str | None = None
    lang: str = Field(default="zh", description="en/zh/es/fr")
    use_rag: bool | None = Field(default=None, description="None=按问题启发式决定")
    use_taobao: bool = False
    order_no: str | None = None
    taobao_method: str | None = Field(
        default=None,
        description="TOP 方法名，如 taobao.trade.fullinfo.get；空则 use_taobao+order_no 自动查单",
    )
    timeout: float | None = None


class CompetitorWarnRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "监控 Temu 上 SKU-BAG-001 的价格",
                    "sku": "SKU-BAG-001",
                    "competitor": "Temu",
                    "via_master": True,
                },
                {
                    "sku": "SKU-BAG-001",
                    "competitor": "Temu",
                    "compete_price": 29.9,
                    "via_master": False,
                },
            ]
        }
    )

    query: str | None = None
    sku: str | None = None
    competitor: str | None = None
    compete_price: float | None = None
    via_master: bool = Field(
        default=True,
        description="True=走 Master 规划；False=直达 data_query",
    )
    timeout: float | None = None


class ApiResult(BaseModel):
    task_id: str
    target: str
    reply_from: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error_msg: str = ""
    msg_type: str = ""
    summary: str = Field(
        default="",
        description="LLM 整理后的可读中文摘要；关闭 OUTPUT_POLISH 时为启发式文案",
    )
    performance: dict[str, Any] = Field(default_factory=dict)
