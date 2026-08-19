"""竞品价格查询：URL 构建 / JSON 解析 / http 模式（mock）。"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ecom_agent_matrix.modules.skills  # noqa: F401
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.modules.skills.competitor_price import (
    _build_adapter_url,
    _extract_price_from_payload,
)


def test_build_adapter_url_with_placeholders():
    url = _build_adapter_url(
        "https://price.internal/v1?sku={sku}&p={competitor}",
        "SKU-BAG-001",
        "Temu",
    )
    assert "SKU-BAG-001" in url and "Temu" in url


def test_build_adapter_url_appends_query():
    url = _build_adapter_url("https://price.internal/v1", "SKU-1", "Amazon")
    assert url.startswith("https://price.internal/v1?")
    assert "sku=SKU-1" in url and "competitor=Amazon" in url


def test_extract_price_nested():
    price, cur, ref = _extract_price_from_payload(
        {"data": {"price": "41.5", "currency": "USD", "source_ref": "https://x"}}
    )
    assert price == 41.5 and cur == "USD" and ref == "https://x"


async def test_http_mode_uses_adapter():
    class _Resp:
        status = 200

        async def text(self):
            return '{"compete_price": 33.3, "currency": "USD", "source_ref": "https://adapter/p"}'

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def get(self, *a, **k):
            return _Resp()

    with patch.object(settings, "COMPETITOR_PRICE_MODE", "http"), patch.object(
        settings, "COMPETITOR_PRICE_API_URL", "https://price.internal/v1"
    ), patch.object(settings, "COMPETITOR_HTTP_FALLBACK_DEMO", False), patch(
        "ecom_agent_matrix.modules.skills.competitor_price.aiohttp.ClientSession",
        _Session,
    ):
        res = await exec_skill(
            "competitor_price",
            {"target_sku": "SKU-BAG-001", "competitor": "Temu"},
        )
    assert res.success is True
    assert res.data["compete_price"] == 33.3
    assert res.data["price_source"] == "http_adapter"


async def test_http_mode_fails_without_fallback():
    with patch.object(settings, "COMPETITOR_PRICE_MODE", "http"), patch.object(
        settings, "COMPETITOR_PRICE_API_URL", ""
    ), patch.object(settings, "COMPETITOR_HTTP_FALLBACK_DEMO", False):
        res = await exec_skill(
            "competitor_price",
            {"target_sku": "SKU-BAG-001", "competitor": "Temu"},
        )
    assert res.success is False
    assert "COMPETITOR_PRICE_API_URL" in (res.error_msg or "")


if __name__ == "__main__":
    test_build_adapter_url_with_placeholders()
    test_build_adapter_url_appends_query()
    test_extract_price_nested()
    asyncio.run(test_http_mode_uses_adapter())
    asyncio.run(test_http_mode_fails_without_fallback())
    print("✅ competitor_price http tests ok")
