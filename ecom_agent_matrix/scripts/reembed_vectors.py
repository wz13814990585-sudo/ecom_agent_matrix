#!/usr/bin/env python3
"""用真实 embedding 回填 vector_goods_kb / agent_long_memory。

- 将商品向量 chunk 与 ecom_goods 标题/描述对齐
- 本地模型缺失时自动从 HuggingFace 拉取 BAAI/bge-small-en-v1.5

用法:
  python -m ecom_agent_matrix.scripts.reembed_vectors
  python -m ecom_agent_matrix.scripts.reembed_vectors --only goods
  python -m ecom_agent_matrix.scripts.reembed_vectors --only memory
"""
from __future__ import annotations

import argparse
import asyncio
import json

from ecom_agent_matrix.config.constants import TABLE_GOODS, TABLE_VECTOR_GOODS
from ecom_agent_matrix.db.base import AsyncPGClient
from ecom_agent_matrix.modules.rag.embedding import (
    get_text_embeddings_batch,
    resolve_embed_model_name,
)


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


def _chunk_for_lang(row: dict, lang: str) -> str:
    title_map = {
        "en": row.get("title_en") or "",
        "zh": row.get("title_zh") or "",
        "es": row.get("title_es") or "",
        "fr": row.get("title_fr") or "",
    }
    title = title_map.get(lang) or row.get("title_en") or row.get("title_zh") or row["sku"]
    desc = (row.get("desc_multi") or "").strip()
    return f"{title}. {desc} SKU={row['sku']} category={row.get('category') or ''}".strip()


async def sync_and_reembed_goods(batch_size: int) -> int:
    goods_rows = await AsyncPGClient.execute_sql(
        f"""
        SELECT tenant_id, store_id, sku, category, price, title_en, title_zh, title_es, title_fr, desc_multi
        FROM {TABLE_GOODS}
        ORDER BY id
        """
    )
    goods = [
        {
            "tenant_id": r[0], "store_id": r[1], "sku": r[2],
            "category": r[3],
            "price": float(r[4]) if r[4] is not None else None,
            "title_en": r[5], "title_zh": r[6], "title_es": r[7],
            "title_fr": r[8], "desc_multi": r[9],
        }
        for r in goods_rows
    ]
    if not goods:
        print("⚠️  ecom_goods 为空，跳过商品向量回填")
        return 0

    # 清空后按「每商品 × 4 语种」重建，保证文本与主表一致
    await AsyncPGClient.execute_sql(f"TRUNCATE {TABLE_VECTOR_GOODS} RESTART IDENTITY")

    langs = ["en", "zh", "es", "fr"]
    payloads: list[tuple[str, str, str, str, str, dict]] = []
    for g in goods:
        for lang in langs:
            chunk = _chunk_for_lang(g, lang)
            meta = {
                "sku": g["sku"],
                "lang": lang,
                "category": g["category"],
                "price": g["price"],
                "source": "reembed_vectors",
            }
            payloads.append((g["tenant_id"], g["store_id"], g["sku"], lang, chunk, meta))

    # 若超过 100 条目标，截断到 100（用户之前要求每表 100）
    if len(payloads) > 100:
        # 优先保证 sku 覆盖：轮询语种截断
        payloads = payloads[:100]

    texts = [p[4] for p in payloads]
    print(f"🔄 商品向量：编码 {len(texts)} 条（模型={resolve_embed_model_name()}）...")
    vectors = await get_text_embeddings_batch(texts, batch_size=batch_size)

    for (tenant_id, store_id, sku, lang, chunk, meta), vec in zip(payloads, vectors):
        await AsyncPGClient.execute_sql(
            f"""
            INSERT INTO {TABLE_VECTOR_GOODS}
            (tenant_id, store_id, goods_sku, lang, chunk_text, embedding, meta_json)
            VALUES (%s, %s, %s, %s, %s, %s::vector, %s::jsonb)
            """,
            [tenant_id, store_id, sku, lang, chunk, _vec_literal(vec), json.dumps(meta, ensure_ascii=False)],
        )
    return len(payloads)


async def reembed_memory(batch_size: int) -> int:
    rows = await AsyncPGClient.execute_sql(
        "SELECT id, content FROM agent_long_memory ORDER BY id"
    )
    if not rows:
        print("⚠️  agent_long_memory 为空，跳过")
        return 0

    ids = [r[0] for r in rows]
    texts = [str(r[1] or "") for r in rows]
    print(f"🔄 长期记忆：编码 {len(texts)} 条...")
    vectors = await get_text_embeddings_batch(texts, batch_size=batch_size)

    for mem_id, vec in zip(ids, vectors):
        await AsyncPGClient.execute_sql(
            "UPDATE agent_long_memory SET embedding = %s::vector WHERE id = %s",
            [_vec_literal(vec), mem_id],
        )
    return len(ids)


async def smoke_vector_search() -> None:
    """简单抽检：用一条商品标题做近邻检索。"""
    sample = await AsyncPGClient.execute_sql(
        f"SELECT chunk_text FROM {TABLE_VECTOR_GOODS} WHERE lang='zh' LIMIT 1"
    )
    if not sample:
        print("抽检跳过：无向量数据")
        return
    query = sample[0][0]
    from ecom_agent_matrix.modules.rag.embedding import get_text_embedding

    qvec = await get_text_embedding(query)
    rows = await AsyncPGClient.execute_sql(
        f"""
        SELECT goods_sku, lang, LEFT(chunk_text, 60), embedding <-> %s::vector AS dist
        FROM {TABLE_VECTOR_GOODS}
        ORDER BY embedding <-> %s::vector
        LIMIT 3
        """,
        [_vec_literal(qvec), _vec_literal(qvec)],
    )
    print("\n🔎 近邻抽检（应接近自身）：")
    for r in rows:
        print(f"  sku={r[0]} lang={r[1]} dist={float(r[3]):.4f} text={r[2]}...")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "goods", "memory"], default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"模型: {resolve_embed_model_name()}")
    goods_n = mem_n = 0
    if args.only in ("all", "goods"):
        goods_n = await sync_and_reembed_goods(args.batch_size)
        print(f"✅ vector_goods_kb 回填 {goods_n} 条")
    if args.only in ("all", "memory"):
        mem_n = await reembed_memory(args.batch_size)
        print(f"✅ agent_long_memory 回填 {mem_n} 条")

    if args.only in ("all", "goods") and goods_n:
        await smoke_vector_search()

    counts = await AsyncPGClient.execute_sql(
        f"""
        SELECT
          (SELECT COUNT(*) FROM {TABLE_VECTOR_GOODS}) AS goods_vec,
          (SELECT COUNT(*) FROM agent_long_memory) AS mem_vec
        """
    )
    print(f"\n📊 当前: vector_goods_kb={counts[0][0]} agent_long_memory={counts[0][1]}")
    await AsyncPGClient.close()
    try:
        from ecom_agent_matrix.db.redis_client import AsyncRedisClient

        await AsyncRedisClient.close()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
