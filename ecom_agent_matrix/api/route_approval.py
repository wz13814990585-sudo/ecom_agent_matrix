"""Authenticated human approval endpoint for high-risk writes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ecom_agent_matrix.api.auth import get_current_security_context
from ecom_agent_matrix.core.security import (
    SecurityContext,
    effective_scopes,
    tenant_scope_from_security,
)
from ecom_agent_matrix.core.security.approval import approval_service
from ecom_agent_matrix.core.security.audit import record_audit_event
from ecom_agent_matrix.platform.resilience.rate_limit import enforce_business_rate_limit

router = APIRouter(prefix="/api/v1/approvals", tags=["approvals"])


@router.post("/{approval_id}/approve", dependencies=[Depends(enforce_business_rate_limit)])
async def approve_request(
    approval_id: str,
    security: SecurityContext = Depends(get_current_security_context),
) -> dict:
    if "risk:approve" not in effective_scopes(security):
        await record_audit_event(
            "AUTHORIZATION_DENIED",
            scope=tenant_scope_from_security(security),
            user_id=security.user_id,
            approval_id=approval_id,
            outcome="denied",
            reason_code="MISSING_RISK_APPROVE_SCOPE",
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="PERMISSION_DENIED")
    try:
        grant = await approval_service.approve(approval_id, security)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Approval not found") from None
    except PermissionError as exc:
        code = str(exc)
        safe_code = code if code in {
            "SELF_APPROVAL_DENIED", "APPROVAL_EXPIRED", "APPROVAL_INVALID"
        } else "APPROVAL_INVALID"
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=safe_code) from None
    return {"approval_id": grant.approval_id, "status": grant.status, "skill_name": grant.skill_name}


__all__ = ["router"]
