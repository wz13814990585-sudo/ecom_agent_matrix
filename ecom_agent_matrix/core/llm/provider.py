"""LLM Provider 抽象：业务只依赖本接口，不依赖具体供应商。"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from ecom_agent_matrix.core.llm.http import get_http_session, with_retry
from ecom_agent_matrix.core.llm.http import is_retryable
from ecom_agent_matrix.core.llm.types import (
    ChatMode,
    ChatResult,
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServerError,
    normalize_mode,
)
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.platform.observability.context import (
    get_trace_context,
    record_llm_usage,
)
from ecom_agent_matrix.platform.observability.metrics import estimate_llm_cost, metrics
from ecom_agent_matrix.platform.resilience.circuit_breaker import get_circuit_breaker

logger = setup_logger("llm.provider")


class LLMProvider(ABC):
    """所有大模型供应商的统一入口。"""

    name: str

    @abstractmethod
    def is_configured(self) -> bool:
        """是否已配置可用的 API Key。"""

    def resolve_mode(self, mode: ChatMode | str | None = None) -> ChatMode:
        return normalize_mode(mode, default=self.default_mode())

    @abstractmethod
    def resolve_model(self, mode: ChatMode | str | None = None) -> str:
        """按模式解析实际 model 名。"""

    def default_mode(self) -> str:
        return "chat"

    @abstractmethod
    async def chat(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.2,
        max_tokens: int = 512,
        mode: ChatMode | str | None = None,
    ) -> ChatResult:
        """调用 Chat Completions。"""


class OpenAICompatProvider(LLMProvider):
    """OpenAI 兼容协议（/chat/completions）的通用实现。

    DeepSeek、OpenAI 及多数兼容网关都走这一层；差异只在配置与少量 payload。
    """

    completions_path: str = "/chat/completions"

    @abstractmethod
    def api_key(self) -> str: ...

    @abstractmethod
    def base_url(self) -> str: ...

    @abstractmethod
    def chat_model(self) -> str: ...

    @abstractmethod
    def reasoner_model(self) -> str: ...

    def key_env_name(self) -> str:
        return "API_KEY"

    def timeout(self) -> float:
        return 60.0

    def max_retries(self) -> int:
        return 2

    def retry_base_delay(self) -> float:
        return 0.8

    def reasoner_min_tokens(self) -> int:
        return 1

    def use_max_completion_tokens(self, mode: ChatMode) -> bool:
        """o-series 等推理模型用 max_completion_tokens。"""
        return False

    def is_configured(self) -> bool:
        return bool((self.api_key() or "").strip())

    def resolve_model(self, mode: ChatMode | str | None = None) -> str:
        resolved = self.resolve_mode(mode)
        if resolved == "reasoner":
            return (self.reasoner_model() or "").strip()
        return (self.chat_model() or "").strip()

    def clamp_max_tokens(self, mode: ChatMode, max_tokens: int) -> int:
        tokens = max(1, int(max_tokens))
        if mode != "reasoner":
            return tokens
        floor = int(self.reasoner_min_tokens() or 1)
        if tokens >= floor:
            return tokens
        logger.info(
            "llm_raise_max_tokens",
            extra={
                "event": "llm_raise_max_tokens",
                "provider": self.name,
                "mode": mode,
                "requested": tokens,
                "applied": floor,
            },
        )
        return floor

    def build_payload(
        self,
        *,
        model: str,
        mode: ChatMode,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """reasoner 通常不支持 temperature 等采样参数。"""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        token_key = (
            "max_completion_tokens"
            if self.use_max_completion_tokens(mode)
            else "max_tokens"
        )
        payload[token_key] = max_tokens
        if mode == "chat":
            payload["temperature"] = temperature
        return payload

    def raise_for_status(self, status: int, body: Any) -> None:
        msg = f"{self.name} API request failed with status {status}"
        if status in (401, 403):
            raise LLMAuthError(msg, status=status, body=body)
        if status == 429:
            raise LLMRateLimitError(msg, status=status, body=body)
        if status >= 500:
            raise LLMServerError(msg, status=status, body=body)
        raise LLMError(msg, status=status, body=body)

    def parse_result(
        self, body: dict[str, Any], fallback_model: str, mode: ChatMode
    ) -> ChatResult:
        try:
            choice0 = body["choices"][0]
            message = choice0["message"]
            content = (message.get("content") or "").strip()
            reasoning = (
                message.get("reasoning_content")
                or message.get("reasoning")
                or choice0.get("reasoning_content")
                or ""
            )
            if isinstance(reasoning, str):
                reasoning = reasoning.strip()
            else:
                reasoning = str(reasoning or "")

            usage = body.get("usage") or {}
            finish_reason = str(choice0.get("finish_reason") or "")

            if not content:
                hint = f"finish_reason={finish_reason or '?'}"
                if reasoning:
                    hint += f", reasoning_len={len(reasoning)}"
                floor = int(self.reasoner_min_tokens() or 1)
                raise LLMResponseError(
                    f"{self.name} 返回空 content（{hint}）。"
                    f"若使用 reasoner，请增大 max_tokens（建议>={floor}）",
                    body=body,
                )

            return ChatResult(
                content=content,
                reasoning_content=reasoning,
                model=str(body.get("model") or fallback_model),
                mode=mode,
                provider=self.name,
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                total_tokens=int(usage.get("total_tokens") or 0),
                raw=body if isinstance(body, dict) else {},
            )
        except LLMResponseError:
            raise
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            raise LLMResponseError(
                f"{self.name} response format is invalid", body=body
            ) from exc

    async def chat(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.2,
        max_tokens: int = 512,
        mode: ChatMode | str | None = None,
    ) -> ChatResult:
        resolved = self.resolve_mode(mode)
        breaker = get_circuit_breaker(
            f"llm:{self.name}",
            failure_threshold=int(settings.CIRCUIT_FAILURE_THRESHOLD),
            reset_seconds=float(settings.CIRCUIT_RESET_SECONDS),
        )
        return await with_retry(
            lambda: breaker.call(lambda: self._chat_once(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                mode=resolved,
            ), is_transient=is_retryable),
            max_retries=int(self.max_retries()),
            base_delay=float(self.retry_base_delay()),
            extra={"provider": self.name, "mode": resolved},
        )

    async def _chat_once(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        mode: ChatMode,
    ) -> ChatResult:
        if not self.is_configured():
            raise LLMAuthError(
                f"未配置 {self.key_env_name()}，请在项目根目录 .env 中设置"
            )

        model = self.resolve_model(mode)
        max_tokens = self.clamp_max_tokens(mode, max_tokens)
        url = f"{self.base_url().rstrip('/')}{self.completions_path}"
        headers = {
            "Authorization": f"Bearer {self.api_key().strip()}",
            "Content-Type": "application/json",
        }
        payload = self.build_payload(
            model=model,
            mode=mode,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        started = time.perf_counter()
        trace = get_trace_context()
        purpose = trace.workflow or {
            "crm_reply": "crm_reply", "ad_optimize": "ad", "ops_report": "report"
        }.get(trace.skill_name, "other")
        if purpose not in {"planner", "recovery", "polish", "rag_answer", "crm_reply", "ad", "report"}:
            purpose = "other"
        session = await get_http_session()
        timeout = aiohttp.ClientTimeout(total=float(self.timeout()))
        try:
            async with session.post(
                url, headers=headers, json=payload, timeout=timeout
            ) as resp:
                try:
                    body = await resp.json(content_type=None)
                except Exception as exc:
                    text = await resp.text()
                    raise LLMResponseError(
                        f"{self.name} response is not JSON (status={resp.status})",
                        status=resp.status,
                        body=text[:500],
                    ) from exc
                if resp.status >= 400:
                    self.raise_for_status(resp.status, body)
                result = self.parse_result(
                    body if isinstance(body, dict) else {}, model, mode
                )
        except asyncio.CancelledError:
            raise
        except (asyncio.TimeoutError, TimeoutError) as exc:
            metrics.observe_llm(self.name, purpose, False, time.perf_counter() - started)
            record_llm_usage(0, 0, 0, None)
            raise LLMServerError(f"{self.name} request timed out", status=504) from exc
        except aiohttp.ClientError as exc:
            metrics.observe_llm(self.name, purpose, False, time.perf_counter() - started)
            record_llm_usage(0, 0, 0, None)
            raise LLMServerError(f"{self.name} dependency unavailable", status=503) from exc
        except Exception:
            metrics.observe_llm(self.name, purpose, False, time.perf_counter() - started)
            record_llm_usage(0, 0, 0, None)
            raise

        latency_ms = (time.perf_counter() - started) * 1000
        estimated_cost = estimate_llm_cost(
            self.name, result.model, result.prompt_tokens, result.completion_tokens,
            settings.LLM_PRICE_TABLE,
        )
        metrics.observe_llm(
            self.name, purpose, True, latency_ms / 1000,
            result.prompt_tokens, result.completion_tokens, estimated_cost,
        )
        record_llm_usage(
            result.prompt_tokens, result.completion_tokens, result.total_tokens, estimated_cost
        )
        logger.info(
            "llm_chat_ok",
            extra={
                "event": "llm_chat_ok",
                "provider": self.name,
                "model": result.model,
                "mode": mode,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "latency_ms": round(latency_ms, 2),
                "has_reasoning": bool(result.reasoning_content),
                "estimated_cost_usd": estimated_cost,
            },
        )
        return result
