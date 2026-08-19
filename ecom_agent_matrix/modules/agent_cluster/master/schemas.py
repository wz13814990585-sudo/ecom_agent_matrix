"""Master typed plan、执行状态与 recovery contract。"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AgentId = Literal["data_query", "biz_exec", "knowledge_rag"]
StepStatus = Literal["PENDING", "RUNNING", "SUCCESS", "FAILED", "SKIPPED"]


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    step_id: str = Field(min_length=2, max_length=64)
    agent: AgentId
    task_type: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    required: bool = True

    @field_validator("step_id")
    @classmethod
    def stable_semantic_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
            raise ValueError("step_id 必须为稳定的 snake_case 语义标识")
        if re.fullmatch(r"step_?\d+", value):
            raise ValueError("step_id 不得使用数组下标语义")
        return value

    @field_validator("depends_on")
    @classmethod
    def unique_dependencies(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("depends_on 不得重复")
        return value


class MasterPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    decision: Literal["execute", "clarify"]
    steps: list[PlanStep] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    reason_code: str = Field(min_length=1, max_length=100)
    clarification_question: str = ""
    planner_source: str = Field(min_length=1)

    @model_validator(mode="after")
    def decision_shape(self) -> "MasterPlan":
        if self.decision == "clarify" and self.steps:
            raise ValueError("clarify plan 不得包含执行步骤")
        if self.decision == "execute" and not self.steps:
            raise ValueError("execute plan 至少需要一个步骤")
        return self


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    agent: AgentId
    task_type: str
    status: StepStatus
    success: bool = False
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: str = ""
    error_msg: str = ""
    correlation_id: str = ""
    latency_ms: float = Field(default=0, ge=0)


class PlanExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_results: dict[str, StepResult] = Field(default_factory=dict)
    all_success: bool
    partial_success: bool
    timed_out: bool
    latency_ms: float = Field(default=0, ge=0)


class RecoveryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["retry_agent", "replan", "finish", "clarify"]
    step_id: str = ""
    reason_code: str = Field(min_length=1)
    final_answer: str = ""
    clarification_question: str = ""


class LLMUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class MasterLLMUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planner: LLMUsage = Field(default_factory=LLMUsage)
    recovery: LLMUsage = Field(default_factory=LLMUsage)
    polish: LLMUsage = Field(default_factory=LLMUsage)
    calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
