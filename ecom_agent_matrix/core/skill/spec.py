"""Skill 的业务契约元数据。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RiskLevel = Literal["low", "medium", "high", "critical"]


class SkillSpec(BaseModel):
    """与具体供应商、存储实现无关的统一 Skill 描述。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str = Field(min_length=1)
    description: str
    version: str = "1.0"
    read_only: bool
    side_effect: bool
    risk_level: RiskLevel
    timeout_seconds: float = Field(default=30.0, gt=0)
    idempotent: bool = False
    input_model: type[BaseModel] | None = None
    output_model: type[BaseModel] | None = None
    deprecated: bool = False
    replacement: str | None = None
    required_scopes: frozenset[str] = Field(default_factory=frozenset)
    approval_required: bool = False

    @model_validator(mode="after")
    def validate_access_metadata(self) -> "SkillSpec":
        if self.read_only and self.side_effect:
            raise ValueError("read_only=True 时 side_effect 必须=False")
        if self.deprecated and not (self.replacement or "").strip():
            raise ValueError("deprecated Skill 必须声明 replacement")
        if self.side_effect and self.risk_level in {"high", "critical"} and not self.approval_required:
            raise ValueError("high/critical side-effect Skill 必须 approval_required=True")
        return self
