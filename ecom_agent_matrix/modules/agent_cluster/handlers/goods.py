"""商品检索 handler：按名搜 SKU，或目录统计/列表。由 Query Agent 调用，不是独立 Agent。"""
import time

from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.skill.skill_registry import exec_skill
from ecom_agent_matrix.modules.skills.goods_catalog import is_catalog_query, wants_full_catalog

logger = setup_logger("agent.goods")


def _extract_product_name(payload: dict) -> str:
    for key in ("product_name", "goods_name", "name", "query", "user_query", "text"):
        val = payload.get(key)
        if val is not None and str(val).strip():
            return _clean_product_query(str(val).strip())
    return ""


def _clean_product_query(text: str) -> str:
    """去掉任务意图噪声，尽量留下商品名（如「防水背包的竞价对比」→「防水背包」）。"""
    import re

    t = str(text or "").strip()
    t = re.sub(
        r"(我想知道|帮我|请|查询|查一下|看看)?",
        "",
        t,
        count=1,
    )
    t = re.sub(
        r"(的)?(竞价对比|价格对比|比价|竞品监控|价格监控|库存|备货|补货).*$",
        "",
        t,
    )
    t = re.sub(r"[的了呢吗啊]+$", "", t.strip())
    return t.strip() or str(text or "").strip()


def _want_catalog(payload: dict) -> bool:
    mode = str(payload.get("mode") or payload.get("goods_mode") or "").strip().lower()
    if mode in {"catalog", "list", "count"}:
        return True
    if str(payload.get("task_type") or "").strip() == "goods_catalog":
        return True
    if payload.get("list_all") or payload.get("catalog"):
        return True
    text = " ".join(
        str(payload.get(k) or "")
        for k in ("query", "user_query", "text", "product_name")
    )
    return is_catalog_query(text)


async def handle_goods(payload: dict) -> tuple[bool, str, dict]:
    """商品目录统计/列表，或按商品名检索候选 SKU。供 Query Agent 调用。"""
    started = time.perf_counter()
    product_name = _extract_product_name(payload)
    top_k = int(payload.get("top_k", 5))
    catalog = _want_catalog(payload)

    if catalog:
        catalog_params: dict = {
            "offset": int(payload.get("offset", 0)),
            "category": payload.get("category"),
            "order_by": payload.get("order_by", "id"),
            "query": product_name,
            "list_all": bool(payload.get("list_all")) or wants_full_catalog(product_name),
            "store_id": payload.get("store_id") or payload.get("scope"),
        }
        if payload.get("limit") is not None:
            catalog_params["limit"] = int(payload["limit"])
        elif payload.get("top_k") is not None and not catalog_params["list_all"]:
            catalog_params["limit"] = int(payload["top_k"])
        skill_result = await exec_skill("goods_catalog", catalog_params)
        data = skill_result.data or {}
        elapsed_ms = (time.perf_counter() - started) * 1000
        return (
            skill_result.success,
            skill_result.error_msg if not skill_result.success else "",
            {
                "query_kind": "goods_catalog",
                "mode": "catalog",
                "product_name": product_name,
                "scope": data.get("scope"),
                "store_id": data.get("store_id"),
                "store_name": data.get("store_name"),
                "is_demo_store": data.get("is_demo_store"),
                "is_external": data.get("is_external"),
                "total": data.get("total", 0),
                "count": data.get("count", 0),
                "items": data.get("items", []),
                "summary": data.get("summary", ""),
                "limit": data.get("limit"),
                "offset": data.get("offset"),
                "list_all": data.get("list_all"),
                "truncated": data.get("truncated"),
                "category": data.get("category"),
                "candidates": [],
                "best_sku": None,
                "latency_ms": round(elapsed_ms, 2),
            },
        )

    if not product_name:
        return (
            False,
            "缺少商品名，请提供 product_name 或 query；查全库请说「有多少商品/列出商品」",
            {"query_kind": "goods_search", "mode": "search", "product_name": "", "candidates": [], "best_sku": None},
        )

    skill_result = await exec_skill(
        "goods_sku_search",
        {"product_name": product_name, "top_k": top_k},
    )
    data = skill_result.data or {}
    elapsed_ms = (time.perf_counter() - started) * 1000
    ok = skill_result.success and bool(data.get("candidates"))
    return (
        ok,
        (
            skill_result.error_msg
            if not skill_result.success
            else ("未找到匹配商品" if not data.get("candidates") else "")
        ),
        {
            "query_kind": "goods_search",
            "mode": "search",
            "product_name": product_name,
            "candidates": data.get("candidates", []),
            "best_sku": data.get("best_sku"),
            "count": data.get("count", 0),
            "match_mode": data.get("match_mode", "none"),
            "semantic_fallback_used": data.get("semantic_fallback_used", False),
            "semantic_error": data.get("semantic_error", ""),
            "latency_ms": round(elapsed_ms, 2),
        },
    )
