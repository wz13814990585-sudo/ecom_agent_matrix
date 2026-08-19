"""商品领域 TaskContext → GoodsRequest 解析。"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ecom_agent_matrix.core.tasking import TaskContext

_CATALOG_HINT = re.compile(
    r"(有多少(?:个)?商品|多少(?:个)?商品|商品数量|商品总数|一共有?(?:多少)?商品|"
    r"列出(?:全部|所有)?商品|有哪些商品|全部商品|所有商品|商品列表|商品目录|"
    r"数据库.*商品|库里.*商品|查(?:询)?.*商品表|商品表|查询数据库|查一下?库|"
    r"数据库里有|库里有什么|看看数据库|how\s+many\s+products?|"
    r"list\s+(?:all\s+)?products?|product\s+catalog|all\s+skus?|"
    r"sku\s+count|count\s+(?:of\s+)?(?:goods|products?)|"
    r"query\s+(?:the\s+)?(?:database|db)|what.?s\s+in\s+(?:the\s+)?(?:database|db))",
    re.IGNORECASE,
)
_FULL_LIST_HINT = re.compile(
    r"(全部|所有|完整|一整[个份]|都列|列全|展示全部|显示全部|"
    r"list\s+all|show\s+all|all\s+products?|entire\s+catalog|full\s+list)",
    re.IGNORECASE,
)
_COUNT_ONLY_HINT = re.compile(
    r"(有多少|多少个|数量|总数|一共|how\s+many|"
    r"count\s+(?:of\s+)?(?:goods|products?)|sku\s+count)",
    re.IGNORECASE,
)


def is_catalog_query(text: str) -> bool:
    """是否为商品目录数量/列表意图。"""
    value = str(text or "")
    return bool(re.search(r"外部站|外部店铺|市场商品", value, re.I) or _CATALOG_HINT.search(value))


def wants_full_catalog(text: str) -> bool:
    """是否明确要求完整商品列表。"""
    value = str(text or "")
    if _FULL_LIST_HINT.search(value):
        return True
    if re.search(r"有哪些商品|列出.*商品|商品列表|商品目录|list\s+products?", value, re.I):
        return not (
            _COUNT_ONLY_HINT.search(value)
            and not re.search(r"哪些|列出|列表|list", value, re.I)
        )
    return False


class GoodsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    mode: Literal["search", "catalog"]
    query: str = ""
    product_name: str | None = None
    top_k: int = Field(default=5, ge=1)
    offset: int = Field(default=0, ge=0)
    limit: int | None = Field(default=None, ge=1)
    category: str | None = None
    order_by: str = "id"
    list_all: bool = False
    store_id: str | None = None


def clean_product_query(text: str) -> str:
    """去掉查询意图噪声，保留尽可能稳定的商品名称。"""
    original = str(text or "").strip()
    cleaned = re.sub(
        r"^(?:(?:我想知道|请帮我|帮我|请|查询|查一下|看看)\s*)+",
        "",
        original,
    )
    cleaned = re.sub(
        r"(的)?(竞价对比|价格对比|比价|竞品监控|价格监控|库存|备货|补货).*$",
        "",
        cleaned,
    )
    cleaned = re.sub(r"[的了呢吗啊]+$", "", cleaned.strip())
    return cleaned.strip() or original


def parse_goods_request(task: TaskContext) -> GoodsRequest:
    """只使用 canonical TaskContext 和真实业务配置生成商品请求。"""
    params = task.params
    product_name = clean_product_query(task.product_name or task.query)
    explicit_mode = str(params.get("mode") or params.get("goods_mode") or "").strip().lower()
    catalog = (
        explicit_mode in {"catalog", "list", "count"}
        or task.task_type == "goods_catalog"
        or bool(params.get("list_all"))
        or bool(params.get("catalog"))
        or is_catalog_query(" ".join(value for value in (task.query, task.product_name) if value))
    )
    list_all = bool(params.get("list_all")) or wants_full_catalog(task.query or product_name)

    top_k = params.get("top_k", 5)
    limit = params.get("limit")
    if limit is None and "top_k" in params and catalog and not list_all:
        limit = top_k

    return GoodsRequest(
        mode="catalog" if catalog else "search",
        query=task.query,
        product_name=product_name or None,
        top_k=top_k,
        offset=params.get("offset", 0),
        limit=limit,
        category=params.get("category"),
        order_by=params.get("order_by") or "id",
        list_all=list_all,
        store_id=task.store_id or params.get("scope"),
    )
