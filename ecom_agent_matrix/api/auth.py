"""接口鉴权：X-API-Key（未配置 API_KEY 时本地放行）。"""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from ecom_agent_matrix.config.settings import settings


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = (settings.API_KEY or "").strip()
    if not expected:
        return
    if not x_api_key or x_api_key.strip() != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或缺失 X-API-Key",
        )
