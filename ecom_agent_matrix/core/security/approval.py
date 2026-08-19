"""Database-backed exact-parameter, tenant-bound, one-time approvals."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.security.approval_models import ApprovalGrant, ApprovalRequest
from ecom_agent_matrix.core.security.audit import record_audit_event
from ecom_agent_matrix.core.security.context import SecurityContext
from ecom_agent_matrix.core.security.policy import effective_scopes
from ecom_agent_matrix.core.security.scope import TenantScope, tenant_scope_from_security
from ecom_agent_matrix.db.base import AsyncPGClient

APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
APPROVAL_EXPIRED = "APPROVAL_EXPIRED"
APPROVAL_ALREADY_USED = "APPROVAL_ALREADY_USED"
APPROVAL_INVALID = "APPROVAL_INVALID"


def approval_params_hash(skill_name: str, validated_params: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"skill_name": skill_name, "params": validated_params},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _request_from_row(row) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=str(row[0]), tenant_id=row[1], store_id=row[2],
        requester_user_id=row[3], approver_user_id=row[4] or "", task_id=row[5],
        skill_name=row[6], params_hash=row[7], status=row[8], requested_at=row[9],
        expires_at=row[10], approved_at=row[11], consumed_at=row[12],
        reason_code=row[13] or "",
    )


class ApprovalService:
    _SELECT = """
      SELECT approval_id,tenant_id,store_id,requester_user_id,approver_user_id,
             task_id,skill_name,params_hash,status,requested_at,expires_at,
             approved_at,consumed_at,reason_code
      FROM security_approval
    """

    async def create_pending(
        self,
        *,
        context,
        skill_name: str,
        params_hash: str,
    ) -> ApprovalRequest:
        scope = TenantScope(
            tenant_id=context.tenant_id, store_id=context.store_id,
            identity_trusted=context.identity_trusted,
        )
        now = datetime.now(timezone.utc)
        request = ApprovalRequest(
            approval_id=str(uuid.uuid4()), task_id=context.task_id,
            tenant_id=scope.tenant_id, store_id=scope.store_id,
            requester_user_id=context.user_id, skill_name=skill_name,
            params_hash=params_hash, status="pending", requested_at=now,
            expires_at=now + timedelta(seconds=int(settings.APPROVAL_TTL_SECONDS)),
        )
        await AsyncPGClient.execute_write(
            """
            INSERT INTO security_approval(
              approval_id,tenant_id,store_id,requester_user_id,task_id,skill_name,
              params_hash,status,requested_at,expires_at
            ) VALUES (%s::uuid,%s,%s,%s,%s,%s,%s,'pending',%s,%s)
            """,
            [request.approval_id, request.tenant_id, request.store_id,
             request.requester_user_id, request.task_id, request.skill_name,
             request.params_hash, request.requested_at, request.expires_at],
            scope=scope,
        )
        await record_audit_event(
            "APPROVAL_REQUESTED", scope=scope, task_id=request.task_id,
            user_id=request.requester_user_id, skill_name=skill_name,
            approval_id=request.approval_id, outcome="pending",
        )
        return request

    async def get_request(self, approval_id: str, security: SecurityContext) -> ApprovalRequest | None:
        scope = tenant_scope_from_security(security)
        # Approval records are internal security state and are intentionally not
        # granted to the general application read role.
        rows = await AsyncPGClient.execute_write(
            self._SELECT + " WHERE approval_id=%s::uuid AND tenant_id=%s AND store_id=%s",
            [approval_id, scope.tenant_id, scope.store_id], scope=scope,
        )
        return _request_from_row(rows[0]) if rows else None

    async def approve(self, approval_id: str, security: SecurityContext) -> ApprovalGrant:
        scope = tenant_scope_from_security(security)
        request = await self.get_request(approval_id, security)
        if request is None:
            raise LookupError("APPROVAL_NOT_FOUND")
        is_admin = "admin" in security.roles
        if request.requester_user_id == security.user_id and not is_admin:
            await record_audit_event(
                "APPROVAL_REJECTED", scope=scope, user_id=security.user_id,
                approval_id=approval_id, skill_name=request.skill_name,
                outcome="rejected", reason_code="SELF_APPROVAL_DENIED",
            )
            raise PermissionError("SELF_APPROVAL_DENIED")
        now = datetime.now(timezone.utc)
        if request.expires_at <= now:
            raise PermissionError(APPROVAL_EXPIRED)
        reason = "ADMIN_SELF_APPROVAL" if is_admin and request.requester_user_id == security.user_id else ""
        rows = await AsyncPGClient.execute_write(
            """
            UPDATE security_approval SET status='approved',approver_user_id=%s,
              approved_at=%s,reason_code=%s
            WHERE approval_id=%s::uuid AND tenant_id=%s AND store_id=%s
              AND status='pending' AND expires_at>%s
            RETURNING approval_id
            """,
            [security.user_id, now, reason, approval_id, scope.tenant_id,
             scope.store_id, now], scope=scope,
        )
        if not rows:
            raise PermissionError(APPROVAL_INVALID)
        await record_audit_event(
            "APPROVAL_APPROVED", scope=scope, task_id=request.task_id,
            user_id=security.user_id, skill_name=request.skill_name,
            approval_id=approval_id, outcome="approved", reason_code=reason,
        )
        return ApprovalGrant(
            approval_id=approval_id, task_id=request.task_id,
            tenant_id=request.tenant_id, store_id=request.store_id,
            requester_user_id=request.requester_user_id,
            approver_user_id=security.user_id, skill_name=request.skill_name,
            params_hash=request.params_hash, expires_at=request.expires_at,
        )

    async def get_grant(self, approval_id: str, security: SecurityContext) -> ApprovalGrant:
        request = await self.get_request(approval_id, security)
        if request is None or request.status != "approved":
            raise LookupError(APPROVAL_INVALID)
        if request.expires_at <= datetime.now(timezone.utc):
            raise PermissionError(APPROVAL_EXPIRED)
        return ApprovalGrant(
            approval_id=request.approval_id, task_id=request.task_id,
            tenant_id=request.tenant_id, store_id=request.store_id,
            requester_user_id=request.requester_user_id,
            approver_user_id=request.approver_user_id,
            skill_name=request.skill_name, params_hash=request.params_hash,
            expires_at=request.expires_at,
        )

    async def consume(self, grant: ApprovalGrant, *, context, skill_name: str, params_hash: str) -> None:
        now = datetime.now(timezone.utc)
        if grant.expires_at <= now:
            raise PermissionError(APPROVAL_EXPIRED)
        if (
            grant.status != "approved" or grant.tenant_id != context.tenant_id
            or grant.store_id != context.store_id or grant.skill_name != skill_name
            or grant.params_hash != params_hash
        ):
            raise PermissionError(APPROVAL_INVALID)
        scope = TenantScope(
            tenant_id=context.tenant_id, store_id=context.store_id,
            identity_trusted=context.identity_trusted,
        )
        rows = await AsyncPGClient.execute_write(
            """
            UPDATE security_approval SET status='consumed',consumed_at=%s
            WHERE approval_id=%s::uuid AND tenant_id=%s AND store_id=%s
              AND status='approved' AND skill_name=%s AND params_hash=%s AND expires_at>%s
            RETURNING approval_id
            """,
            [now, grant.approval_id, scope.tenant_id, scope.store_id,
             skill_name, params_hash, now], scope=scope,
        )
        if not rows:
            raise PermissionError(APPROVAL_ALREADY_USED)
        await record_audit_event(
            "APPROVAL_CONSUMED", scope=scope, task_id=context.task_id,
            user_id=context.user_id, agent_id=context.agent_id, skill_name=skill_name,
            approval_id=grant.approval_id, outcome="consumed",
        )


approval_service = ApprovalService()

__all__ = [
    "APPROVAL_ALREADY_USED", "APPROVAL_EXPIRED", "APPROVAL_INVALID",
    "APPROVAL_REQUIRED", "ApprovalService", "approval_params_hash", "approval_service",
]
