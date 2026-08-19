#!/usr/bin/env python3
"""将业务/向量相关表补齐到至少 TARGET 条（默认 100）。

用法（项目根目录）:
  python -m ecom_agent_matrix.scripts.seed_demo_data
  python -m ecom_agent_matrix.scripts.seed_demo_data --target 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from ecom_agent_matrix.config.constants import (
    AGENT_EXEC,
    AGENT_MASTER,
    AGENT_QUERY,
    AGENT_RAG,
    TABLE_COMPETITOR,
    TABLE_FINETUNE_DATA,
    TABLE_GOODS,
    TABLE_ORDER,
    TABLE_RISK_LOG,
    TABLE_VECTOR_GOODS,
)
from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.db.base import AsyncPGClient

TARGET_DEFAULT = 100
EMBED_DIM = 384

CATEGORIES = [
    ("bags", "背包", "Bag", "Mochila", "Sac"),
    ("apparel", "服饰", "Dress", "Vestido", "Robe"),
    ("footwear", "鞋履", "Shoes", "Zapatos", "Chaussures"),
    ("accessories", "配件", "Accessory", "Accesorio", "Accessoire"),
    ("camping", "露营", "Camping", "Camping", "Camping"),
    ("electronics", "数码", "Gadget", "Gadget", "Gadget"),
    ("home", "家居", "Home", "Hogar", "Maison"),
    ("sports", "运动", "Sports", "Deportes", "Sport"),
]

PRODUCT_STEMS = [
    ("Waterproof", "防水", "Impermeable", "Imperméable"),
    ("Lightweight", "轻量", "Ligero", "Léger"),
    ("Premium", "高端", "Premium", "Premium"),
    ("Travel", "旅行", "Viaje", "Voyage"),
    ("Outdoor", "户外", "Exterior", "Plein air"),
    ("Compact", "便携", "Compacto", "Compact"),
    ("Eco", "环保", "Eco", "Éco"),
    ("Pro", "专业", "Pro", "Pro"),
    ("Classic", "经典", "Clásico", "Classique"),
    ("Ultra", "超轻", "Ultra", "Ultra"),
]

COMPETITORS = [
    "Amazon", "Temu", "AliExpress", "Shein", "Walmart",
    "eBay", "Shopee", "Lazada", "Decathlon", "REI",
]

# 外部站模拟店铺（与本店 demo_store 区分）
EXTERNAL_STORES = [
    ("ext_amazon", "外部站·Amazon模拟", "Amazon"),
    ("ext_temu", "外部站·Temu模拟", "Temu"),
    ("ext_aliexpress", "外部站·AliExpress模拟", "AliExpress"),
    ("ext_shein", "外部站·Shein模拟", "Shein"),
    ("ext_walmart", "外部站·Walmart模拟", "Walmart"),
    ("ext_ebay", "外部站·eBay模拟", "eBay"),
    ("ext_shopee", "外部站·Shopee模拟", "Shopee"),
    ("ext_lazada", "外部站·Lazada模拟", "Lazada"),
    ("ext_decathlon", "外部站·Decathlon模拟", "Decathlon"),
    ("ext_rei", "外部站·REI模拟", "REI"),
]

EXTERNAL_TARGET_DEFAULT = 1000

RISK_TYPES = [
    ("refund_abuse", "短时间内重复退款申请"),
    ("address_mismatch", "收货地址与账单地址不一致"),
    ("high_velocity", "同一账号短时间高频下单"),
    ("amount_outlier", "单笔金额远超历史均值"),
    ("device_fraud", "设备指纹异常，疑似刷单"),
    ("coupon_stack", "异常叠加优惠券"),
]

TASK_TYPES = ["goods_text", "chat", "social", "tool_call"]
LANGS = ["en", "zh", "es", "fr"]
AGENTS = [
    AGENT_MASTER, AGENT_QUERY, AGENT_EXEC, AGENT_RAG,
]


def _rand_unit_vec(dim: int = EMBED_DIM) -> list[float]:
    vals = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [round(v / norm, 6) for v in vals]


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


async def _count(table: str) -> int:
    rows = await AsyncPGClient.execute_sql(f"SELECT COUNT(*) FROM {table}")
    return int(rows[0][0])


async def _count_goods(*, store_id: str | None = None, external_only: bool = False) -> int:
    if external_only:
        rows = await AsyncPGClient.execute_sql(
            f"SELECT COUNT(*) FROM {TABLE_GOODS} WHERE store_id LIKE 'ext_%%'"
        )
    elif store_id:
        rows = await AsyncPGClient.execute_sql(
            f"SELECT COUNT(*) FROM {TABLE_GOODS} WHERE store_id = %s",
            [store_id],
        )
    else:
        rows = await AsyncPGClient.execute_sql(f"SELECT COUNT(*) FROM {TABLE_GOODS}")
    return int(rows[0][0])


async def _ensure_vector_tables() -> None:
    sql_path = Path(__file__).resolve().parents[1] / "db" / "vector_tables.sql"
    text = sql_path.read_text(encoding="utf-8")
    # 简单按分号拆分执行（跳过空/注释）
    for part in text.split(";"):
        stmt = "\n".join(
            ln for ln in part.splitlines()
            if ln.strip() and not ln.strip().startswith("--")
        ).strip()
        if not stmt:
            continue
        try:
            await AsyncPGClient.execute_sql(stmt)
        except Exception as exc:
            err = str(exc).lower()
            if "already exists" in err:
                continue
            raise


async def _existing_skus(*, own_store_only: bool = True) -> list[str]:
    store_id = settings.DEMO_STORE_ID or "demo_store"
    if own_store_only:
        rows = await AsyncPGClient.execute_sql(
            f"SELECT sku FROM {TABLE_GOODS} WHERE store_id = %s ORDER BY id",
            [store_id],
        )
    else:
        rows = await AsyncPGClient.execute_sql(f"SELECT sku FROM {TABLE_GOODS} ORDER BY id")
    return [r[0] for r in rows]


async def ensure_goods_store_columns() -> None:
    """给已有 ecom_goods 补齐店铺标记字段，并把空值打成模拟店。"""
    alters = [
        f"ALTER TABLE {TABLE_GOODS} ADD COLUMN IF NOT EXISTS store_id VARCHAR(64) DEFAULT 'demo_store'",
        f"ALTER TABLE {TABLE_GOODS} ADD COLUMN IF NOT EXISTS store_name VARCHAR(128) DEFAULT '我的模拟独立站'",
        f"ALTER TABLE {TABLE_GOODS} ADD COLUMN IF NOT EXISTS is_demo BOOLEAN DEFAULT true",
    ]
    for stmt in alters:
        try:
            await AsyncPGClient.execute_sql(stmt)
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise
    store_id = settings.DEMO_STORE_ID or "demo_store"
    store_name = settings.DEMO_STORE_NAME or "我的模拟独立站"
    await AsyncPGClient.execute_sql(
        f"""
        UPDATE {TABLE_GOODS}
        SET store_id = COALESCE(NULLIF(store_id, ''), %s),
            store_name = COALESCE(NULLIF(store_name, ''), %s),
            is_demo = COALESCE(is_demo, true)
        WHERE store_id IS NULL OR store_id = '' OR store_name IS NULL OR store_name = ''
           OR is_demo IS NULL
        """,
        [store_id, store_name],
    )
    # 演示种子统一打标为本模拟店
    await AsyncPGClient.execute_sql(
        f"""
        UPDATE {TABLE_GOODS}
        SET store_id = %s, store_name = %s, is_demo = true
        WHERE sku LIKE 'SKU-%%'
        """,
        [store_id, store_name],
    )


async def seed_goods(target: int) -> int:
    await ensure_goods_store_columns()
    store_id = settings.DEMO_STORE_ID or "demo_store"
    store_name = settings.DEMO_STORE_NAME or "我的模拟独立站"
    n = await _count_goods(store_id=store_id)
    need = max(0, target - n)
    if need == 0:
        return 0
    for i in range(need):
        idx = n + i + 1
        cat_key, cat_zh, cat_en, cat_es, cat_fr = CATEGORIES[idx % len(CATEGORIES)]
        stem_en, stem_zh, stem_es, stem_fr = PRODUCT_STEMS[idx % len(PRODUCT_STEMS)]
        sku = f"SKU-{cat_key.upper()[:4]}-{idx:03d}"
        price = round(9.9 + (idx % 40) * 3.5 + random.random() * 2, 2)
        stock = 20 + (idx * 7) % 300
        title_en = f"{stem_en} {cat_en} #{idx}"
        title_zh = f"{stem_zh}{cat_zh} #{idx}"
        title_es = f"{stem_es} {cat_es} #{idx}"
        title_fr = f"{stem_fr} {cat_fr} #{idx}"
        desc = (
            f"{title_en}. Durable {cat_key} for travel and daily use. "
            f"SKU {sku}. Lightweight and easy to pack. "
            f"Store: {store_name} ({store_id})."
        )
        await AsyncPGClient.execute_sql(
            f"""
            INSERT INTO {TABLE_GOODS}
            (sku, category, price, stock_num, title_en, title_zh, title_es, title_fr,
             desc_multi, store_id, store_name, is_demo)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (sku) DO NOTHING
            """,
            [
                sku, cat_key, price, stock, title_en, title_zh, title_es, title_fr, desc,
                store_id, store_name, True,
            ],
        )
    return need


async def seed_external_goods(target: int) -> int:
    """生成外部站模拟货盘（与本店 store_id 分离）。"""
    await ensure_goods_store_columns()
    n = await _count_goods(external_only=True)
    need = max(0, target - n)
    if need == 0:
        return 0
    for i in range(need):
        idx = n + i + 1
        store_id, store_name, platform = EXTERNAL_STORES[idx % len(EXTERNAL_STORES)]
        cat_key, cat_zh, cat_en, cat_es, cat_fr = CATEGORIES[idx % len(CATEGORIES)]
        stem_en, stem_zh, stem_es, stem_fr = PRODUCT_STEMS[idx % len(PRODUCT_STEMS)]
        sku = f"EXT-{platform.upper()[:4]}-{idx:04d}"
        # 外部站价格相对本店略有浮动，便于后续比价演示
        price = round(8.5 + (idx % 45) * 3.2 + random.random() * 4, 2)
        stock = 5 + (idx * 11) % 500
        title_en = f"[{platform}] {stem_en} {cat_en} Ext#{idx}"
        title_zh = f"[{platform}]{stem_zh}{cat_zh} 外部#{idx}"
        title_es = f"[{platform}] {stem_es} {cat_es} Ext#{idx}"
        title_fr = f"[{platform}] {stem_fr} {cat_fr} Ext#{idx}"
        desc = (
            f"External marketplace listing on {platform}. "
            f"{title_en}. Simulated competitor catalog item. "
            f"store_id={store_id}."
        )
        await AsyncPGClient.execute_sql(
            f"""
            INSERT INTO {TABLE_GOODS}
            (sku, category, price, stock_num, title_en, title_zh, title_es, title_fr,
             desc_multi, store_id, store_name, is_demo)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (sku) DO NOTHING
            """,
            [
                sku, cat_key, price, stock, title_en, title_zh, title_es, title_fr, desc,
                store_id, store_name, True,
            ],
        )
    return need


async def seed_orders(target: int) -> int:
    skus = await _existing_skus()
    if not skus:
        return 0
    n = await _count(TABLE_ORDER)
    need = max(0, target - n)
    base = datetime.now() - timedelta(days=90)
    for i in range(need):
        idx = n + i + 1
        sku = skus[idx % len(skus)]
        buy_num = 1 + (idx % 5)
        unit = 19.9 + (idx % 30) * 2.5
        total = round(unit * buy_num, 2)
        order_no = f"ORD-SEED-{idx:04d}"
        refund = (idx % 11 == 0)
        created = base + timedelta(hours=idx * 3)
        await AsyncPGClient.execute_sql(
            f"""
            INSERT INTO {TABLE_ORDER}
            (order_no, sku, buy_num, total_amount, refund_flag, create_time)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (order_no) DO NOTHING
            """,
            [order_no, sku, buy_num, total, refund, created],
        )
    return need


async def seed_competitor(target: int) -> int:
    skus = await _existing_skus()
    if not skus:
        return 0
    n = await _count(TABLE_COMPETITOR)
    need = max(0, target - n)
    base = datetime.now() - timedelta(days=60)
    for i in range(need):
        idx = n + i + 1
        sku = skus[idx % len(skus)]
        competitor = COMPETITORS[idx % len(COMPETITORS)]
        price = round(15 + (idx % 50) * 1.7 + random.random(), 2)
        crawled = base + timedelta(hours=idx * 5)
        await AsyncPGClient.execute_sql(
            f"""
            INSERT INTO {TABLE_COMPETITOR}
            (target_sku, competitor_name, compete_price, crawl_time)
            VALUES (%s, %s, %s, %s)
            """,
            [sku, competitor, price, crawled],
        )
    return need


async def seed_risk(target: int) -> int:
    n = await _count(TABLE_RISK_LOG)
    need = max(0, target - n)
    base = datetime.now() - timedelta(days=45)
    for i in range(need):
        idx = n + i + 1
        risk_type, desc = RISK_TYPES[idx % len(RISK_TYPES)]
        order_no = f"ORD-SEED-{(idx % max(target, 1)) + 1:04d}"
        created = base + timedelta(hours=idx * 4)
        await AsyncPGClient.execute_sql(
            f"""
            INSERT INTO {TABLE_RISK_LOG}
            (order_no, risk_type, risk_desc, create_time)
            VALUES (%s, %s, %s, %s)
            """,
            [order_no, risk_type, f"{desc}（样本 #{idx}）", created],
        )
    return need


async def seed_finetune(target: int) -> int:
    n = await _count(TABLE_FINETUNE_DATA)
    need = max(0, target - n)
    for i in range(need):
        idx = n + i + 1
        task = TASK_TYPES[idx % len(TASK_TYPES)]
        lang = LANGS[idx % len(LANGS)]
        if task == "goods_text":
            inp = f"Write a short title for product sample #{idx}"
            out = f"Premium Outdoor Gear #{idx}"
        elif task == "chat":
            inp = f"这款商品 #{idx} 适合什么场景？"
            out = f"适合日常出行与轻户外场景，样本答复 #{idx}。"
        elif task == "social":
            inp = f"Generate IG caption for product #{idx}"
            out = f"Adventure ready #{idx} ✨ Shop the look today!"
        else:
            inp = f"查询 SKU 样本 #{idx} 库存"
            out = f"stock_num={50 + idx % 200}"
        await AsyncPGClient.execute_sql(
            f"""
            INSERT INTO {TABLE_FINETUNE_DATA}
            (task_type, input_text, output_text, lang)
            VALUES (%s, %s, %s, %s)
            """,
            [task, inp, out, lang],
        )
    return need


async def seed_mcp(target: int) -> int:
    n = await _count("mcp_message_log")
    need = max(0, target - n)
    for i in range(need):
        idx = n + i + 1
        sender = AGENTS[idx % len(AGENTS)]
        target_agent = AGENTS[(idx + 3) % len(AGENTS)]
        if target_agent == sender:
            target_agent = AGENTS[(idx + 1) % len(AGENTS)]
        task_id = f"seed-mcp-{idx:04d}-{uuid.uuid4().hex[:8]}"
        content = {
            "query": f"seed task #{idx}",
            "sku": f"SKU-BAG-{(idx % 20) + 1:03d}",
            "demo": True,
        }
        await AsyncPGClient.execute_sql(
            """
            INSERT INTO mcp_message_log
            (task_id, sender_agent, target_agent, priority, msg_content)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            """,
            [task_id, sender, target_agent, idx % 5, json.dumps(content, ensure_ascii=False)],
        )
    return need


async def seed_vector_goods(target: int) -> int:
    skus = await _existing_skus()
    if not skus:
        return 0
    n = await _count(TABLE_VECTOR_GOODS)
    need = max(0, target - n)
    for i in range(need):
        idx = n + i + 1
        sku = skus[idx % len(skus)]
        lang = LANGS[idx % len(LANGS)]
        chunk = (
            f"[{lang}] Product knowledge for {sku}. "
            f"Feature highlight sample #{idx}: waterproof, lightweight, travel-ready."
        )
        meta = {"sku": sku, "lang": lang, "seed": True, "idx": idx}
        vec = _vec_literal(_rand_unit_vec())
        await AsyncPGClient.execute_sql(
            f"""
            INSERT INTO {TABLE_VECTOR_GOODS}
            (goods_sku, lang, chunk_text, embedding, meta_json)
            VALUES (%s, %s, %s, %s::vector, %s::jsonb)
            """,
            [sku, lang, chunk, vec, json.dumps(meta, ensure_ascii=False)],
        )
    return need


async def seed_long_memory(target: int) -> int:
    n = await _count("agent_long_memory")
    need = max(0, target - n)
    for i in range(need):
        idx = n + i + 1
        agent = AGENTS[idx % len(AGENTS)]
        content = (
            f"记忆样本 #{idx} agent={agent}: "
            f"历史任务处理摘要，SKU 偏好与阈值记录。"
        )
        meta = {
            "seed": True,
            "confidence": round(0.6 + (idx % 40) / 100, 2),
            "success": True,
            "idx": idx,
        }
        vec = _vec_literal(_rand_unit_vec())
        await AsyncPGClient.execute_sql(
            """
            INSERT INTO agent_long_memory
            (agent_name, content, embedding, meta_json)
            VALUES (%s, %s, %s::vector, %s::jsonb)
            """,
            [agent, content, vec, json.dumps(meta, ensure_ascii=False)],
        )
    return need


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed demo DB rows")
    parser.add_argument("--target", type=int, default=TARGET_DEFAULT, help="本店及相关表目标条数")
    parser.add_argument(
        "--external-target",
        type=int,
        default=EXTERNAL_TARGET_DEFAULT,
        help="外部站商品目标条数（默认 1000）",
    )
    args = parser.parse_args()
    target = max(1, int(args.target))
    external_target = max(0, int(args.external_target))

    print(f"🎯 本店及相关表目标：至少 {target} 条")
    print(f"🎯 外部站商品目标：至少 {external_target} 条")
    await _ensure_vector_tables()

    own_added = await seed_goods(target)
    ext_added = await seed_external_goods(external_target) if external_target else 0

    inserted = {
        f"{TABLE_GOODS}(本店)": own_added,
        f"{TABLE_GOODS}(外部站)": ext_added,
        TABLE_ORDER: await seed_orders(target),
        TABLE_COMPETITOR: await seed_competitor(target),
        TABLE_RISK_LOG: await seed_risk(target),
        TABLE_FINETUNE_DATA: await seed_finetune(target),
        "mcp_message_log": await seed_mcp(target),
        TABLE_VECTOR_GOODS: await seed_vector_goods(target),
        "agent_long_memory": await seed_long_memory(target),
    }

    print("\n📦 本次新增：")
    for k, v in inserted.items():
        print(f"  {k}: +{v}")

    store_id = settings.DEMO_STORE_ID or "demo_store"
    print("\n📊 当前总量：")
    print(f"  ecom_goods 本店({store_id}): {await _count_goods(store_id=store_id)}")
    print(f"  ecom_goods 外部站(ext_*): {await _count_goods(external_only=True)}")
    print(f"  ecom_goods 合计: {await _count(TABLE_GOODS)}")
    for table in (
        TABLE_ORDER,
        TABLE_COMPETITOR,
        TABLE_RISK_LOG,
        TABLE_FINETUNE_DATA,
        "mcp_message_log",
        TABLE_VECTOR_GOODS,
        "agent_long_memory",
    ):
        print(f"  {table}: {await _count(table)}")

    await AsyncPGClient.close()
    print("\n✅ 种子数据完成")


if __name__ == "__main__":
    asyncio.run(main())
