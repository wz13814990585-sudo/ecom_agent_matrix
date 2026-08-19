"""Typed high-risk approval envelopes; no raw business payloads are stored."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    approval_id: str
    task_id: str
    tenant_id: str
    store_id: str
    requester_user_id: str
    skill_name: str
    params_hash: str
    status: Literal["pending", "approved", "rejected", "consumed", "expired"]
    requested_at: datetime
    expires_at: datetime
    approver_user_id: str = ""
    approved_at: datetime | None = None
    consumed_at: datetime | None = None
    reason_code: str = ""


class ApprovalGrant(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    approval_id: str
    task_id: str
    tenant_id: str
    store_id: str
    requester_user_id: str
    approver_user_id: str
    skill_name: str
    params_hash: str
    status: Literal["approved"] = "approved"
    expires_at: datetime


__all__ = ["ApprovalGrant", "ApprovalRequest"]
