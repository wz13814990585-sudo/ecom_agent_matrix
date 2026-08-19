"""客服 workflow：best-effort 会话记忆、可选淘宝查询与直接目标语种答复。"""
from __future__ import annotations

import time

from pydantic import ValidationError
from ecom_agent_matrix.platform.observability.metrics import observed_workflow

from ecom_agent_matrix.core.memory.short_memory import AgentShortMemory
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.core.tasking import TaskContext, WorkflowResult, ensure_task_context
from ecom_agent_matrix.core.tasking.result import INVALID_REQUEST, PARTIAL_SUCCESS, SKILL_FAILED
from ecom_agent_matrix.modules.parsers.crm import CRMRequest, parse_crm_request
from ecom_agent_matrix.modules.rag.formatter import format_rag_context, normalize_documents
from ecom_agent_matrix.modules.rag.policy import should_retrieve_knowledge
from ecom_agent_matrix.modules.rag.schemas import RAGRequest
from ecom_agent_matrix.modules.rag.service import rag_service
from ecom_agent_matrix.core.security import tenant_scope_from_task_context


def _metadata(started: float, **extra) -> dict:
    return {
        "workflow": "crm",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        **extra,
    }


def _safe_fallback(request: CRMRequest) -> str:
    if request.lang == "zh":
        return "客服答复服务暂不可用，请稍后再试，或提供订单号以便人工跟进。"
    return "Customer support is temporarily unavailable. Please try again or provide your order number."


async def _run_taobao(request: CRMRequest) -> tuple[dict, str]:
    method = request.taobao_method
    payload = dict(request.taobao_payload)
    if not method and request.use_taobao and request.order_no:
        method = "taobao.trade.fullinfo.get"
        payload = {
            "tid": request.order_no,
            "fields": "tid,status,total_fee,payment,orders",
        }
    if not method:
        return {"skipped": True}, ""
    result = await exec_skill("taobao_api", {"method": method, "payload": payload})
    return (
        {
            "skipped": False,
            "success": result.success,
            "error_code": result.error_code,
            "error_msg": result.error_msg,
            "data": result.data or {},
            "method": method,
        },
        "" if result.success else (result.error_code or SKILL_FAILED),
    )


def _upstream_knowledge(request: CRMRequest) -> tuple[str, list[dict], int]:
    for value in request.upstream_context.values():
        if not (
            isinstance(value, dict)
            and value.get("task_type") == "knowledge_qa"
            and isinstance(value.get("data"), dict)
        ):
            continue
        data = value["data"]
        citations = [
            item for item in (data.get("citations") or []) if isinstance(item, dict)
        ][:20]
        raw_docs = data.get("documents") or data.get("docs") or []
        documents = (
            normalize_documents([item for item in raw_docs if isinstance(item, dict)])
            if isinstance(raw_docs, list)
            else []
        )
        context = format_rag_context(documents) if documents else ""
        if not context:
            context = str(data.get("answer") or data.get("summary") or "").strip()[:5000]
        if context:
            return context, citations, len(documents) or len(citations)
    return "", [], 0


@observed_workflow("crm")
async def run_crm_workflow(
    task: dict | TaskContext,
    *,
    task_id: str = "",
) -> WorkflowResult:
    started = time.perf_counter()
    if isinstance(task, TaskContext):
        ctx = task
    else:
        ctx = ensure_task_context(task)
        if task_id:
            ctx = ctx.with_updates(task_id=task_id.strip())
    try:
        request = parse_crm_request(ctx)
    except (ValidationError, TypeError, ValueError) as exc:
        return WorkflowResult(
            success=False,
            error_code=INVALID_REQUEST,
            error_msg=f"客服请求参数不合法：{exc}",
            data={"exec_kind": "crm"},
            metadata=_metadata(started),
        )
    if not request.query:
        return WorkflowResult(
            success=False,
            error_code=INVALID_REQUEST,
            error_msg="user_query 为空",
            data={"exec_kind": "crm", "session_id": request.session_id},
            metadata=_metadata(started),
        )

    memory_errors: list[str] = []
    history: list = []
    short_memory: AgentShortMemory | None = None
    try:
        memory_identity = (
            {"tenant_id": ctx.tenant_id, "user_id": ctx.user_id}
            if ctx.identity_trusted and ctx.tenant_id and ctx.user_id
            else {}
        )
        short_memory = AgentShortMemory(
            session_id=request.session_id,
            **memory_identity,
        )
    except Exception as exc:
        memory_errors.append(f"init:{type(exc).__name__}")
    if short_memory is not None:
        try:
            await short_memory.append(role="user", content=request.query)
        except Exception as exc:
            memory_errors.append(f"append_user:{type(exc).__name__}")
        try:
            history = await short_memory.get_all()
        except Exception as exc:
            memory_errors.append(f"read:{type(exc).__name__}")
            history = []

    taobao_info, taobao_error_code = await _run_taobao(request)
    knowledge_context, citations, rag_doc_count = _upstream_knowledge(request)
    rag_error = ""
    has_policy_context = bool(knowledge_context)
    should_retrieve = (
        not has_policy_context
        and not request.is_fallback_route
        and should_retrieve_knowledge(request.query, request.use_rag)
    )
    if should_retrieve:
        retrieval = await rag_service.retrieve(
            RAGRequest(
                query=request.query,
                lang=request.lang,
                top_k=5,
                task_id=request.task_id,
            ),
            scope=tenant_scope_from_task_context(ctx),
        )
        if retrieval.success:
            knowledge_context = format_rag_context(retrieval.documents)
            citations = [citation.model_dump() for citation in retrieval.citations]
            rag_doc_count = retrieval.recall_count
        else:
            rag_error = retrieval.error_code or "RETRIEVAL_ERROR"
    reply_result = await exec_skill(
        "crm_reply",
        {
            "user_query": request.query,
            "lang": request.lang,
            "history": history,
            "use_rag": False if has_policy_context else request.use_rag,
            "taobao_info": taobao_info,
            "is_fallback_route": request.is_fallback_route,
            "task_id": request.task_id,
            "upstream_context": request.upstream_context,
            "knowledge_context": knowledge_context,
            "citations": citations,
        },
    )
    reply_data = reply_result.data or {}
    answer = str(reply_data.get("answer") or "").strip()
    if not answer and not reply_result.success:
        answer = _safe_fallback(request).strip()
    if not answer:
        return WorkflowResult(
            success=False,
            error_code=SKILL_FAILED,
            error_msg=reply_result.error_msg or "crm_reply 未返回可用答复",
            data={
                "exec_kind": "crm",
                "session_id": request.session_id,
                "answer": "",
                "lang": request.lang,
            },
            metadata=_metadata(
                started,
                skill_error_code=reply_result.error_code,
                memory_errors=memory_errors,
            ),
        )

    if short_memory is not None:
        try:
            await short_memory.append(role="assistant", content=answer)
        except Exception as exc:
            memory_errors.append(f"append_assistant:{type(exc).__name__}")

    llm_ok = bool(reply_data.get("llm_ok"))
    rag_used = bool(knowledge_context)
    rag_doc_count = int(rag_doc_count or reply_data.get("rag_doc_count") or 0)
    rag_error = rag_error or str(reply_data.get("rag_error") or "")
    errors: list[str] = []
    skill_error_codes: dict[str, str] = {}
    if taobao_error_code:
        errors.append(f"taobao: {taobao_info.get('error_msg') or 'failed'}")
        skill_error_codes["taobao_api"] = taobao_error_code
    if not reply_result.success:
        errors.append(f"crm_reply: {reply_result.error_msg or 'failed'}")
        skill_error_codes["crm_reply"] = reply_result.error_code or SKILL_FAILED
    elif not llm_ok:
        errors.append("crm_reply_used_fallback")
    if rag_error:
        errors.append(f"rag: {rag_error}")
    errors.extend(f"memory: {error}" for error in memory_errors)
    partial = bool(errors)
    return WorkflowResult(
        success=True,
        partial_success=partial,
        error_code=PARTIAL_SUCCESS if partial else "",
        error_msg="; ".join(errors),
        data={
            "exec_kind": "crm",
            "session_id": request.session_id,
            "answer": answer,
            "lang": request.lang,
            "llm_ok": llm_ok,
            "translate_ok": True,
            "trans_info": {
                "skipped": True,
                "reason": "crm_reply handles target language directly",
            },
            "rag_used": rag_used,
            "rag_doc_count": rag_doc_count,
            "citations": citations,
            "taobao": taobao_info,
            "partial_success": partial,
        },
        metadata=_metadata(
            started,
            memory_errors=memory_errors,
            skill_error_codes=skill_error_codes,
        ),
    )


async def handle_crm(
    task: dict | TaskContext,
    *,
    task_id: str = "",
) -> tuple[bool, str, dict]:
    return (await run_crm_workflow(task, task_id=task_id)).as_legacy_tuple()
