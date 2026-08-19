"""商品名 → SKU 检索：字面/pg_trgm 优先，向量语义兜底。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ecom_agent_matrix.config.constants import TABLE_GOODS, TABLE_VECTOR_GOODS
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.skill.skill_registry import register_skill
from ecom_agent_matrix.db.base import AsyncPGClient
from ecom_agent_matrix.modules.rag.embedding import get_text_embedding


class GoodsSkuSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    product_name: str | None = None
    query: str | None = None
    name: str | None = None
    top_k: int = Field(default=5, ge=1)
    force_semantic: bool = False

    @model_validator(mode="after")
    def require_search_text(self) -> "GoodsSkuSearchInput":
        if not any((self.product_name, self.query, self.name)):
            raise ValueError("必须提供 product_name、query 或 name")
        return self


class GoodsSkuSearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str
    candidates: list[dict[str, Any]]
    best_sku: str | None
    count: int = Field(ge=0)
    match_mode: str
    semantic_fallback_used: bool
    semantic_error: str


def _row_to_candidate(
    row,
    *,
    match_mode: str,
    score: float | None = None,
    dist: float | None = None,
) -> dict:
    item = {
        "sku": row[0],
        "title_zh": row[1],
        "title_en": row[2],
        "category": row[3],
        "price": float(row[4]) if row[4] is not None else None,
        "stock_num": row[5],
        "match_mode": match_mode,
    }
    if score is not None:
        item["trgm_score"] = round(float(score), 4)
    if dist is not None:
        item["vector_dist"] = round(float(dist), 4)
        item["vector_score"] = round(1.0 / (1.0 + float(dist)), 4)
    return item


async def _literal_ilike_only(product_name: str, top_k: int) -> list[dict]:
    """无 pg_trgm 时的降级：纯 ILIKE（大数据量会慢，仅兜底）。"""
    pattern = f"%{product_name}%"
    sql = f"""
    SELECT sku, title_zh, title_en, category, price, stock_num
    FROM {TABLE_GOODS}
    WHERE title_zh ILIKE %s
       OR title_en ILIKE %s
       OR title_es ILIKE %s
       OR title_fr ILIKE %s
       OR sku ILIKE %s
       OR COALESCE(desc_multi, '') ILIKE %s
    ORDER BY stock_num DESC NULLS LAST
    LIMIT %s
    """
    rows = await AsyncPGClient.execute_sql(
        sql, [pattern, pattern, pattern, pattern, pattern, pattern, top_k]
    )
    return [_row_to_candidate(r, match_mode="literal_ilike") for r in rows]


async def _literal_trgm_search(product_name: str, top_k: int) -> list[dict]:
    """
    字面模糊 + pg_trgm：
    - ILIKE '%x%' 在 gin_trgm_ops 上可走索引，避免大表顺序扫
    - %% 相似度操作符覆盖轻度错字/截断
    """
    pattern = f"%{product_name}%"
    min_sim = float(settings.GOODS_SEARCH_TRGM_MIN_SIM)
    sql = f"""
    SELECT sku, title_zh, title_en, category, price, stock_num,
           GREATEST(
             similarity(COALESCE(title_zh, ''), %s),
             similarity(COALESCE(title_en, ''), %s),
             similarity(COALESCE(title_es, ''), %s),
             similarity(COALESCE(title_fr, ''), %s),
             similarity(COALESCE(sku, ''), %s),
             similarity(COALESCE(desc_multi, ''), %s)
           ) AS sim
    FROM {TABLE_GOODS}
    WHERE title_zh ILIKE %s
       OR title_en ILIKE %s
       OR title_es ILIKE %s
       OR title_fr ILIKE %s
       OR sku ILIKE %s
       OR COALESCE(desc_multi, '') ILIKE %s
       OR title_zh %% %s
       OR title_en %% %s
       OR title_es %% %s
       OR title_fr %% %s
       OR sku %% %s
    ORDER BY sim DESC, stock_num DESC NULLS LAST
    LIMIT %s
    """
    params = [
        product_name, product_name, product_name, product_name, product_name, product_name,
        pattern, pattern, pattern, pattern, pattern, pattern,
        product_name, product_name, product_name, product_name, product_name,
        max(top_k * 3, top_k),
    ]
    rows = await AsyncPGClient.execute_sql(sql, params)
    candidates = []
    for r in rows:
        sim = float(r[6] or 0)
        # ILIKE 子串命中：即便 similarity 偏低也保留
        text_blob = f"{r[0]} {r[1]} {r[2]}".lower()
        ilike_hit = product_name.lower() in text_blob
        if not ilike_hit and sim < min_sim:
            continue
        candidates.append(_row_to_candidate(r, match_mode="literal_trgm", score=sim))
        if len(candidates) >= top_k:
            break
    return candidates


async def _semantic_vector_search(product_name: str, top_k: int) -> list[dict]:
    """
    向量语义兜底：同义词 / 口语（如「防雨登山包」→ 防水户外背包）。
    依赖 vector_goods_kb，按 SKU 聚合后 LEFT JOIN ecom_goods hydrate。
    """
    query_vec = await get_text_embedding(product_name)
    max_dist = float(settings.GOODS_SEARCH_VECTOR_MAX_DIST)
    recall_k = max(int(settings.GOODS_SEARCH_VECTOR_TOP_K), top_k)

    sql = f"""
    SELECT
        COALESCE(g.sku, v.goods_sku) AS sku,
        g.title_zh,
        g.title_en,
        COALESCE(g.category, v.meta_json->>'category') AS category,
        g.price,
        g.stock_num,
        MIN(v.embedding <-> %s::vector) AS dist
    FROM {TABLE_VECTOR_GOODS} v
    LEFT JOIN {TABLE_GOODS} g ON g.sku = v.goods_sku
    GROUP BY COALESCE(g.sku, v.goods_sku), g.title_zh, g.title_en,
             COALESCE(g.category, v.meta_json->>'category'), g.price, g.stock_num
    HAVING MIN(v.embedding <-> %s::vector) <= %s
    ORDER BY dist ASC
    LIMIT %s
    """
    rows = await AsyncPGClient.execute_sql(sql, [query_vec, query_vec, max_dist, recall_k])
    return [
        _row_to_candidate(r, match_mode="semantic_vector", dist=float(r[6]))
        for r in rows[:top_k]
        if r[0]
    ]


def _merge_candidates(primary: list[dict], secondary: list[dict], top_k: int) -> list[dict]:
    """字面优先，语义补齐，按 sku 去重。"""
    seen = set()
    merged: list[dict] = []
    for item in primary + secondary:
        sku = item.get("sku")
        if not sku or sku in seen:
            continue
        seen.add(sku)
        merged.append(item)
        if len(merged) >= top_k:
            break
    return merged


@register_skill
class GoodsSkuSearchTool(BaseSkill):
    read_only = True
    side_effect = False
    risk_level = "low"
    timeout_seconds = 15.0
    idempotent = True
    input_model = GoodsSkuSearchInput
    output_model = GoodsSkuSearchOutput
    skill_name = "goods_sku_search"
    skill_desc = (
        "根据中文/英文商品名查候选 SKU："
        "优先 pg_trgm/ILIKE 字面模糊，无结果时向量语义兜底；"
        "参数 product_name、top_k、force_semantic"
    )

    async def run(self, params: dict) -> SkillResult:
        try:
            product_name = str(
                params.get("product_name") or params.get("query") or params.get("name") or ""
            ).strip()
            top_k = int(params.get("top_k", 5))
            force_semantic = bool(params.get("force_semantic", False))

            if not product_name:
                return SkillResult(success=False, error_msg="缺少 product_name / query")
            if top_k <= 0:
                return SkillResult(success=False, error_msg="top_k 必须为正整数")

            literal_hits: list[dict] = []
            semantic_hits: list[dict] = []
            match_mode = "none"
            semantic_error = ""
            literal_mode = "literal_trgm"

            if not force_semantic:
                try:
                    literal_hits = await _literal_trgm_search(product_name, top_k)
                except Exception as exc:
                    err = str(exc).lower()
                    if any(k in err for k in ("similarity", "pg_trgm", "operator does not exist")):
                        literal_hits = await _literal_ilike_only(product_name, top_k)
                        literal_mode = "literal_ilike"
                    else:
                        raise

            need_semantic = force_semantic or (
                not literal_hits and settings.GOODS_SEARCH_SEMANTIC_FALLBACK
            )
            if need_semantic:
                try:
                    semantic_hits = await _semantic_vector_search(product_name, top_k)
                except Exception as exc:
                    semantic_error = type(exc).__name__
                    if force_semantic:
                        return SkillResult(
                            success=False,
                            error_msg=f"向量语义检索失败：{type(exc).__name__}",
                        )

            candidates = _merge_candidates(literal_hits, semantic_hits, top_k)
            if literal_hits and semantic_hits:
                match_mode = "hybrid"
            elif literal_hits:
                match_mode = literal_mode
            elif semantic_hits:
                match_mode = "semantic_vector"

            return SkillResult(
                success=True,
                data={
                    "product_name": product_name,
                    "candidates": candidates,
                    "best_sku": candidates[0]["sku"] if candidates else None,
                    "count": len(candidates),
                    "match_mode": match_mode,
                    "semantic_fallback_used": bool(need_semantic and semantic_hits),
                    "semantic_error": semantic_error,
                },
            )
        except ValueError:
            return SkillResult(success=False, error_msg="top_k 必须为整数")
        except Exception as exc:
            return SkillResult(success=False, error_msg=f"商品检索异常：{type(exc).__name__}")
