from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from ecom_agent_matrix.api.route_approval import approve_request
from ecom_agent_matrix.core.security import ApprovalGrant, ApprovalRequest, SecurityContext
from ecom_agent_matrix.core.security.approval import (
    APPROVAL_ALREADY_USED,
    APPROVAL_EXPIRED,
    APPROVAL_INVALID,
    ApprovalService,
    approval_params_hash,
)
from ecom_agent_matrix.core.skill.skill_registry import SkillExecutionContext


APPROVAL_ID = "00000000-0000-0000-0000-000000000001"


def _security(user="operator", role="risk_operator", tenant="tenant-a", store="store-a"):
    return SecurityContext(
        subject=user, user_id=user, tenant_id=tenant, store_id=store,
        roles=frozenset({role}), scopes=frozenset(), auth_type="jwt", authenticated=True,
    )


def _context():
    return SkillExecutionContext(
        agent_id="biz_exec", task_id="task-1", tenant_id="tenant-a", store_id="store-a",
        user_id="operator", roles=frozenset({"risk_operator"}), identity_trusted=True,
    )


def _request(**updates):
    now = datetime.now(timezone.utc)
    values = {
        "approval_id": APPROVAL_ID, "task_id": "task-1", "tenant_id": "tenant-a",
        "store_id": "store-a", "requester_user_id": "operator",
        "skill_name": "record_order_risk", "params_hash": "a" * 64,
        "status": "pending", "requested_at": now, "expires_at": now + timedelta(minutes=5),
    }
    values.update(updates)
    return ApprovalRequest(**values)


def _grant(**updates):
    values = {
        "approval_id": APPROVAL_ID, "task_id": "task-1", "tenant_id": "tenant-a",
        "store_id": "store-a", "requester_user_id": "operator",
        "approver_user_id": "approver", "skill_name": "record_order_risk",
        "params_hash": "a" * 64, "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
    }
    values.update(updates)
    return ApprovalGrant(**values)


def test_pending_approval_persists_hash_but_not_raw_params():
    captured = {}

    async def write(sql, params, **kwargs):
        captured.update(sql=sql, params=params, scope=kwargs["scope"])
        return []

    async def scenario():
        with patch(
            "ecom_agent_matrix.core.security.approval.AsyncPGClient.execute_write", new=write
        ), patch(
            "ecom_agent_matrix.core.security.approval.record_audit_event", new=AsyncMock()
        ):
            return await ApprovalService().create_pending(
                context=_context(), skill_name="record_order_risk",
                params_hash=approval_params_hash("record_order_risk", {"order_no": "SECRET-ORDER"}),
            )

    request = asyncio.run(scenario())
    assert request.status == "pending"
    assert "SECRET-ORDER" not in repr(captured["params"])
    assert request.params_hash in captured["params"]
    assert "raw" not in captured["sql"].lower()


def test_risk_operator_cannot_call_approval_endpoint():
    async def scenario():
        with patch(
            "ecom_agent_matrix.api.route_approval.record_audit_event", new=AsyncMock()
        ) as audit:
            with pytest.raises(HTTPException) as raised:
                await approve_request(APPROVAL_ID, _security())
        return raised.value, audit

    error, audit = asyncio.run(scenario())
    assert error.status_code == 403 and error.detail == "PERMISSION_DENIED"
    audit.assert_awaited_once()


def test_risk_approver_can_approve_and_cross_tenant_is_not_disclosed():
    service = AsyncMock()
    service.approve.return_value = _grant()

    async def scenario():
        with patch("ecom_agent_matrix.api.route_approval.approval_service", service):
            approved = await approve_request(APPROVAL_ID, _security("approver", "risk_approver"))
            service.approve.side_effect = LookupError("other tenant")
            with pytest.raises(HTTPException) as raised:
                await approve_request(APPROVAL_ID, _security("other", "risk_approver", "tenant-b"))
        return approved, raised.value

    approved, hidden = asyncio.run(scenario())
    assert approved["status"] == "approved"
    assert hidden.status_code == 404 and hidden.detail == "Approval not found"


def test_self_approval_rejected_but_admin_override_is_audited():
    service = ApprovalService()
    pending = _request()

    async def scenario():
        with patch.object(service, "get_request", new=AsyncMock(return_value=pending)), patch(
            "ecom_agent_matrix.core.security.approval.AsyncPGClient.execute_write",
            new=AsyncMock(return_value=[[APPROVAL_ID]]),
        ), patch(
            "ecom_agent_matrix.core.security.approval.record_audit_event", new=AsyncMock()
        ) as audit:
            with pytest.raises(PermissionError, match="SELF_APPROVAL_DENIED"):
                await service.approve(APPROVAL_ID, _security())
            grant = await service.approve(APPROVAL_ID, _security("operator", "admin"))
        return grant, audit

    grant, audit = asyncio.run(scenario())
    assert grant.approver_user_id == "operator"
    assert any(
        call.args[0] == "APPROVAL_APPROVED"
        and call.kwargs["reason_code"] == "ADMIN_SELF_APPROVAL"
        for call in audit.await_args_list
    )


def test_one_time_consume_is_atomic_and_concurrent_replay_fails():
    service = ApprovalService()
    state = {"approved": True}

    async def write(_sql, _params, **_kwargs):
        if state["approved"]:
            state["approved"] = False
            return [[APPROVAL_ID]]
        return []

    async def consume_once():
        try:
            await service.consume(
                _grant(), context=_context(), skill_name="record_order_risk", params_hash="a" * 64
            )
            return "ok"
        except PermissionError as exc:
            return str(exc)

    async def scenario():
        with patch(
            "ecom_agent_matrix.core.security.approval.AsyncPGClient.execute_write", new=write
        ), patch(
            "ecom_agent_matrix.core.security.approval.record_audit_event", new=AsyncMock()
        ):
            return await asyncio.gather(consume_once(), consume_once())

    assert sorted(asyncio.run(scenario())) == [APPROVAL_ALREADY_USED, "ok"]


@pytest.mark.parametrize(
    "grant,skill_name,params_hash,code",
    [
        (_grant(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)), "record_order_risk", "a" * 64, APPROVAL_EXPIRED),
        (_grant(), "other_skill", "a" * 64, APPROVAL_INVALID),
        (_grant(), "record_order_risk", "b" * 64, APPROVAL_INVALID),
        (_grant(tenant_id="tenant-b"), "record_order_risk", "a" * 64, APPROVAL_INVALID),
    ],
)
def test_expired_wrong_skill_params_or_tenant_is_rejected_before_db(grant, skill_name, params_hash, code):
    write = AsyncMock()

    async def scenario():
        with patch(
            "ecom_agent_matrix.core.security.approval.AsyncPGClient.execute_write", new=write
        ):
            await ApprovalService().consume(
                grant, context=_context(), skill_name=skill_name, params_hash=params_hash
            )

    with pytest.raises(PermissionError, match=code):
        asyncio.run(scenario())
    write.assert_not_awaited()

