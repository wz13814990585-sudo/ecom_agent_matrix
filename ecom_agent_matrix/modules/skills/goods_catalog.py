"""商品目录查询：统计数量 / 列表（支持「全部」拉全量，有上限保护）。"""
from __future__ import annotations

import re

from ecom_agent_matrix.config.constants import TABLE_GOODS
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill
from ecom_agent_matrix.db.base import AsyncPGClient

# 单次最多返回条数，防止响应过大
MAX_CATALOG_LIMIT = 500
DEFAULT_PAGE_LIMIT = 20

_CATALOG_HINT = re.compile(
    r"("
    r"有多少(?:个)?商品|多少(?:个)?商品|商品数量|商品总数|一共有?(?:多少)?商品|"
    r"列出(?:全部|所有)?商品|有哪些商品|全部商品|所有商品|商品列表|商品目录|"
    r"数据库.*商品|库里.*商品|查(?:询)?.*商品表|商品表|"
    r"查询数据库|查一下?库|数据库里有|库里有什么|看看数据库|"
    r"how\s+many\s+products?|list\s+(?:all\s+)?products?|product\s+catalog|"
    r"all\s+skus?|sku\s+count|count\s+(?:of\s+)?(?:goods|products?)|"
    r"query\s+(?:the\s+)?(?:database|db)|what.?s\s+in\s+(?:the\s+)?(?:database|db)"
    r")",
    re.IGNORECASE,
)

_FULL_LIST_HINT = re.compile(
    r"("
    r"全部|所有|完整|一整[个份]|都列|列全|展示全部|显示全部|"
    r"list\s+all|show\s+all|all\s+products?|entire\s+catalog|full\s+list"
    r")",
    re.IGNORECASE,
)

_COUNT_ONLY_HINT = re.compile(
    r"("
    r"有多少|多少个|数量|总数|一共|"
    r"how\s+many|count\s+(?:of\s+)?(?:goods|products?)|sku\s+count"
    r")",
    re.IGNORECASE,
)


def resolve_catalog_scope(text: str = "", store_id: str | None = None) -> tuple[str, str]:
    """
    返回 (scope, store_id_or_filter)。
    scope: own | external | all | platform
    """
    raw = str(store_id or "").strip()
    t = str(text or "")
    if raw == "*" or re.search(r"全部店铺|所有店铺|整个库|全库商品", t, re.I):
        return "all", "*"
    if raw.startswith("ext_"):
        return "platform", raw
    platform_map = {
        "amazon": "ext_amazon",
        "temu": "ext_temu",
        "aliexpress": "ext_aliexpress",
        "shein": "ext_shein",
        "walmart": "ext_walmart",
        "ebay": "ext_ebay",
        "shopee": "ext_shopee",
        "lazada": "ext_lazada",
        "decathlon": "ext_decathlon",
        "rei": "ext_rei",
    }
    for name, sid in platform_map.items():
        if re.search(rf"\b{name}\b", t, re.I) and re.search(r"外部|竞品|市场|站", t):
            return "platform", sid
        if re.search(rf"外部站.*{name}|{name}.*外部", t, re.I):
            return "platform", sid
    if raw.lower() in {"external", "ext", "market"}:
        return "external", "external"
    if re.search(r"外部站|外部店铺|竞品站|市场商品|他站|第三方站", t, re.I):
        return "external", "external"
    if raw and raw != settings.DEMO_STORE_ID:
        return "platform", raw
    return "own", (settings.DEMO_STORE_ID or "demo_store")


def is_catalog_query(text: str) -> bool:
    """是否为「查全库数量/列表」意图（区别于按商品名搜索）。"""
    t = str(text or "")
    if re.search(r"外部站|外部店铺|市场商品", t, re.I):
        return True
    return bool(_CATALOG_HINT.search(t))


def wants_full_catalog(text: str) -> bool:
    """用户是否明确要求看全部商品（而非默认分页预览）。"""
    t = str(text or "")
    if _FULL_LIST_HINT.search(t):
        return True
    # 「有哪些商品 / 列出商品 / 商品列表」视为要列表；若同时只问数量则不算全量
    if re.search(r"有哪些商品|列出.*商品|商品列表|商品目录|list\s+products?", t, re.I):
        if _COUNT_ONLY_HINT.search(t) and not re.search(r"哪些|列出|列表|list", t, re.I):
            return False
        return True
    return False


@register_skill
class GoodsCatalogTool(BaseSkill):
    read_only = True
    side_effect = False
    risk_level = "low"
    skill_name = "goods_catalog"
    skill_desc = (
        "商品目录查询：统计 ecom_goods 数量并列出；"
        "参数 limit、offset、category、order_by、list_all"
    )

    async def run(self, params: dict) -> SkillResult:
        try:
            offset = int(params.get("offset", 0))
            category = str(params.get("category") or "").strip() or None
            order_by = str(params.get("order_by") or "id").strip().lower()
            list_all = bool(params.get("list_all"))
            query_text = str(
                params.get("query")
                or params.get("user_query")
                or params.get("product_name")
                or ""
            )
            if not list_all:
                list_all = wants_full_catalog(query_text)

            if "limit" in params and params.get("limit") is not None:
                limit = int(params.get("limit"))
            elif "top_k" in params and params.get("top_k") is not None:
                limit = int(params.get("top_k"))
            elif list_all:
                limit = MAX_CATALOG_LIMIT
            else:
                limit = DEFAULT_PAGE_LIMIT

            if limit <= 0 or limit > MAX_CATALOG_LIMIT:
                return SkillResult(
                    success=False,
                    error_msg=f"limit 需在 1~{MAX_CATALOG_LIMIT}",
                )
            if offset < 0:
                return SkillResult(success=False, error_msg="offset 不能为负")

            order_sql = {
                "stock": "stock_num DESC NULLS LAST, id DESC",
                "price": "price DESC NULLS LAST, id DESC",
                "id": "id DESC",
            }.get(order_by, "id DESC")

            where = "WHERE 1=1"
            args: list = []
            scope, store_filter = resolve_catalog_scope(
                query_text,
                str(params.get("store_id") or params.get("scope") or "").strip() or None,
            )
            own_name = str(settings.DEMO_STORE_NAME or "我的模拟独立站")
            if scope == "all":
                pass
            elif scope == "external":
                where += " AND store_id LIKE 'ext_%%'"
            elif scope == "platform":
                where += " AND store_id = %s"
                args.append(store_filter)
            else:
                where += " AND store_id = %s"
                args.append(store_filter or settings.DEMO_STORE_ID or "demo_store")

            if category:
                where += " AND LOWER(category) = LOWER(%s)"
                args.append(category)

            count_sql = f"SELECT COUNT(*) FROM {TABLE_GOODS} {where}"
            count_rows = await AsyncPGClient.execute_sql(count_sql, list(args))
            total = int(count_rows[0][0] or 0) if count_rows else 0

            # 要全部时：一次取 min(total, MAX)，从 offset=0
            if list_all:
                limit = min(total, MAX_CATALOG_LIMIT) if total > 0 else limit
                offset = 0

            list_sql = f"""
            SELECT sku, title_zh, title_en, category, price, stock_num,
                   store_id, store_name, is_demo
            FROM {TABLE_GOODS}
            {where}
            ORDER BY {order_sql}
            LIMIT %s OFFSET %s
            """
            list_args = list(args) + [limit, offset]
            rows = await AsyncPGClient.execute_sql(list_sql, list_args)
            items = [
                {
                    "sku": r[0],
                    "title_zh": r[1],
                    "title_en": r[2],
                    "category": r[3],
                    "price": float(r[4]) if r[4] is not None else None,
                    "stock_num": r[5],
                    "store_id": r[6],
                    "store_name": r[7],
                    "is_demo": bool(r[8]) if r[8] is not None else True,
                }
                for r in rows
            ]

            truncated = total > len(items) + offset
            if scope == "all":
                label = "全库（本店+外部站）"
            elif scope == "external":
                label = "外部站模拟货盘"
            elif scope == "platform":
                label = items[0]["store_name"] if items else store_filter
            else:
                label = own_name
            summary = f"【{label}】共 {total} 件（模拟数据）"
            if category:
                summary += f"；类目 {category}"
            if items:
                if list_all and not truncated:
                    summary += f"；已返回全部 {len(items)} 件"
                elif list_all and truncated:
                    summary += (
                        f"；已返回前 {len(items)} 件"
                        f"（单次上限 {MAX_CATALOG_LIMIT}，其余请分页 offset）"
                    )
                else:
                    preview = "、".join(
                        (it.get("title_zh") or it.get("title_en") or it["sku"])
                        for it in items[:5]
                    )
                    summary += f"；本页展示 {len(items)} 件（默认分页），例如：{preview}"
            else:
                summary += "；无数据"

            return SkillResult(
                success=True,
                data={
                    "mode": "catalog",
                    "scope": scope,
                    "store_id": None if store_filter == "*" else store_filter,
                    "store_name": label,
                    "is_demo_store": scope == "own",
                    "is_external": scope in {"external", "platform"},
                    "total": total,
                    "count": len(items),
                    "limit": limit,
                    "offset": offset,
                    "list_all": list_all,
                    "truncated": truncated,
                    "category": category,
                    "order_by": order_by if order_by in ("stock", "price", "id") else "id",
                    "items": items,
                    "summary": summary,
                },
            )
        except ValueError:
            return SkillResult(success=False, error_msg="limit/offset 必须为整数")
        except Exception as exc:
            return SkillResult(success=False, error_msg=f"商品目录查询失败：{exc}")
