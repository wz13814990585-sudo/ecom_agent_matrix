"""Phase 2C-2B CRM parser / workflow tests。"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, patch

from ecom_agent_matrix.core.skill.base_skill import SkillResult
from ecom_agent_matrix.core.tasking import normalize_task_context
from ecom_agent_matrix.core.tasking.result import PARTIAL_SUCCESS, SKILL_FAILED
from ecom_agent_matrix.modules.agent_cluster.handlers import crm as crm_handler
from ecom_agent_matrix.modules.agent_cluster.handlers.crm import handle_crm, run_crm_workflow
from ecom_agent_matrix.modules.parsers.crm import parse_crm_request


def _memory():
    memory = AsyncMock()
    memory.append.return_value = 1
    memory.get_all.return_value = []
    return memory


def _reply(*, llm_ok=True, success=True, answer="answer"):
    return SkillResult(success=success, error_code="EXECUTION_ERROR" if not success else "", error_msg="failed" if not success else "", data={
        "answer": answer, "llm_ok": llm_ok, "rag_used": False, "rag_doc_count": 0, "rag_error": "",
    })


def test_crm_canonical_fields_and_task_session_fallback():
    explicit = parse_crm_request(normalize_task_context(
        {"query": "hello", "lang": "en", "session_id": "S-1"}, task_id="ROOT"
    ))
    fallback = parse_crm_request(normalize_task_context({"query": "你好"}, task_id="ROOT"))
    assert (explicit.query, explicit.lang, explicit.session_id) == ("hello", "en", "S-1")
    assert fallback.session_id == "ROOT"
    assert fallback.lang == "zh"


def test_crm_payload_fake_task_id_cannot_override_envelope():
    request = parse_crm_request(normalize_task_context(
        {"query": "hello", "task_id": "FAKE"}, task_id="REAL"
    ))
    assert request.task_id == "REAL"
    assert request.session_id == "REAL"


def test_crm_order_no_extraction():
    request = parse_crm_request(normalize_task_context({"query": "check ORD-2026-ABC please"}))
    assert request.order_no == "ORD-2026-ABC"


def test_crm_no_longer_calls_text_translate_and_passes_language_to_reply():
    calls = []
    async def skills(name, params):
        calls.append((name, params))
        return _reply()
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.crm.AgentShortMemory", return_value=_memory()), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.exec_skill", side_effect=skills
        ):
            return await run_crm_workflow({"query": "refund please", "lang": "en", "session_id": "S"})
    result = asyncio.run(scenario())
    assert result.success is True
    assert [name for name, _ in calls] == ["crm_reply"]
    assert calls[0][1]["lang"] == "en"
    assert result.data["trans_info"]["skipped"] is True
    assert "text_translate" not in inspect.getsource(crm_handler)


def test_crm_short_memory_read_failure_is_best_effort():
    memory = _memory()
    memory.get_all.side_effect = RuntimeError("redis down")
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.crm.AgentShortMemory", return_value=memory), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.exec_skill", new=AsyncMock(return_value=_reply())
        ):
            return await run_crm_workflow({"query": "hello", "session_id": "S"})
    result = asyncio.run(scenario())
    assert result.success is True
    assert any("read" in error for error in result.metadata["memory_errors"])


def test_crm_short_memory_append_failure_is_best_effort():
    memory = _memory()
    memory.append.side_effect = RuntimeError("redis down")
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.crm.AgentShortMemory", return_value=memory), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.exec_skill", new=AsyncMock(return_value=_reply())
        ):
            return await run_crm_workflow({"query": "hello", "session_id": "S"})
    result = asyncio.run(scenario())
    assert result.success is True
    assert result.metadata["memory_errors"]


def test_crm_taobao_failure_and_reply_success_is_partial():
    async def skills(name, params):
        if name == "taobao_api":
            return SkillResult(success=False, error_code="TIMEOUT", error_msg="timeout")
        return _reply()
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.crm.AgentShortMemory", return_value=_memory()), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.exec_skill", side_effect=skills
        ):
            return await run_crm_workflow({"query": "check ORD-1", "use_taobao": True, "order_no": "ORD-1"})
    result = asyncio.run(scenario())
    assert result.success and result.partial_success
    assert result.error_code == PARTIAL_SUCCESS
    assert result.metadata["skill_error_codes"]["taobao_api"] == "TIMEOUT"


def test_crm_reply_fallback_is_structured_degradation():
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.crm.AgentShortMemory", return_value=_memory()), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.exec_skill", new=AsyncMock(return_value=_reply(llm_ok=False))
        ):
            return await run_crm_workflow({"query": "hello"})
    result = asyncio.run(scenario())
    assert result.success and result.partial_success
    assert result.error_code == PARTIAL_SUCCESS


def test_crm_total_reply_failure_without_fallback_fails():
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.crm.AgentShortMemory", return_value=_memory()), patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.exec_skill", new=AsyncMock(return_value=_reply(success=False, answer=""))
        ), patch("ecom_agent_matrix.modules.agent_cluster.handlers.crm._safe_fallback", return_value=""):
            return await run_crm_workflow({"query": "hello"})
    result = asyncio.run(scenario())
    assert result.success is False
    assert result.error_code == SKILL_FAILED


def test_crm_legacy_task_id_compatibility():
    async def scenario():
        with patch("ecom_agent_matrix.modules.agent_cluster.handlers.crm.AgentShortMemory", return_value=_memory()) as memory_cls, patch(
            "ecom_agent_matrix.modules.agent_cluster.handlers.crm.exec_skill", new=AsyncMock(return_value=_reply())
        ):
            result = await handle_crm({"query": "hello", "task_id": "FAKE"}, task_id="ROOT")
        return result, memory_cls
    (ok, _, data), memory_cls = asyncio.run(scenario())
    assert ok and data["session_id"] == "ROOT"
    memory_cls.assert_called_once_with(session_id="ROOT")
