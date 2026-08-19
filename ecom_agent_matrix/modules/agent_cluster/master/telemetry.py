"""Master Provider invocation budget 与真实 token usage。"""
from __future__ import annotations

from typing import Literal

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.llm import ChatResult
from ecom_agent_matrix.modules.agent_cluster.master.schemas import (
    LLMUsage,
    MasterLLMUsage,
)

UsageKind = Literal["planner", "recovery", "polish"]


class MasterLLMTelemetry:
    def __init__(self, max_calls: int | None = None):
        self.max_calls = int(
            max_calls if max_calls is not None else settings.MASTER_MAX_LLM_CALLS
        )
        self._usage = {
            "planner": LLMUsage(),
            "recovery": LLMUsage(),
            "polish": LLMUsage(),
        }

    @property
    def calls(self) -> int:
        return sum(item.calls for item in self._usage.values())

    def can_call(self) -> bool:
        return self.calls < max(self.max_calls, 0)

    def start_call(self, kind: UsageKind) -> bool:
        """只在真正准备调用 Provider 时计一次 invocation。"""
        if not self.can_call():
            return False
        current = self._usage[kind]
        self._usage[kind] = current.model_copy(update={"calls": current.calls + 1})
        return True

    def add_result(self, kind: UsageKind, result: ChatResult) -> None:
        current = self._usage[kind]
        self._usage[kind] = current.model_copy(
            update={
                "prompt_tokens": current.prompt_tokens + int(result.prompt_tokens or 0),
                "completion_tokens": current.completion_tokens
                + int(result.completion_tokens or 0),
                "total_tokens": current.total_tokens + int(result.total_tokens or 0),
            }
        )

    def snapshot(self) -> MasterLLMUsage:
        prompt_tokens = sum(item.prompt_tokens for item in self._usage.values())
        completion_tokens = sum(item.completion_tokens for item in self._usage.values())
        total_tokens = sum(item.total_tokens for item in self._usage.values())
        return MasterLLMUsage(
            planner=self._usage["planner"],
            recovery=self._usage["recovery"],
            polish=self._usage["polish"],
            calls=self.calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
