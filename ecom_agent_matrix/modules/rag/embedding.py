"""向量化、重排模型封装。"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from ecom_agent_matrix.config.settings import settings
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.db.redis_client import AsyncRedisClient

_embed_model = None
CACHE_TTL = 3600
# 本地路径不存在时回退到 HuggingFace Hub
_HF_FALLBACK = "BAAI/bge-small-en-v1.5"
logger = setup_logger("rag.embedding")


def resolve_embed_model_name() -> str:
    raw = (settings.EMBED_MODEL_PATH or "").strip() or _HF_FALLBACK
    path = Path(raw)
    if path.exists():
        return str(path)
    # 形如 org/name 的 Hub 模型名
    if "/" in raw and not raw.startswith("."):
        return raw
    return _HF_FALLBACK


def get_embed_model() -> Any:
    """延迟加载重型推理依赖，避免 Agent 注册/规划测试启动模型运行时。"""
    global _embed_model
    if _embed_model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        name = resolve_embed_model_name()
        _embed_model = SentenceTransformer(name, device=device)
    return _embed_model


async def get_text_embedding(text: str) -> list[float]:
    """获取文本向量，优先读 Redis 缓存。"""
    model_identity = resolve_embed_model_name()
    cache_key = _embedding_cache_key(text, model_identity)
    redis = None
    try:
        redis = await AsyncRedisClient.get_client()
        cache_data = await redis.get(cache_key)
        if cache_data:
            decoded = json.loads(cache_data)
            if isinstance(decoded, list):
                return decoded
    except Exception as exc:
        logger.warning(
            "embedding_cache_read_failed",
            extra={"event": "embedding_cache_read_failed", "error_type": type(exc).__name__},
        )
    model = await asyncio.wait_for(
        asyncio.to_thread(get_embed_model), timeout=float(settings.EMBEDDING_TIMEOUT_SECONDS)
    )
    vec = await asyncio.wait_for(
        asyncio.to_thread(model.encode, text), timeout=float(settings.EMBEDDING_TIMEOUT_SECONDS)
    )
    vec = vec.tolist()
    try:
        if redis is None:
            redis = await AsyncRedisClient.get_client()
        await redis.set(cache_key, json.dumps(vec), ex=CACHE_TTL)
    except Exception as exc:
        logger.warning(
            "embedding_cache_write_failed",
            extra={"event": "embedding_cache_write_failed", "error_type": type(exc).__name__},
        )
    return vec


def _embedding_cache_key(text: str, model_identity: str | None = None) -> str:
    identity = model_identity or resolve_embed_model_name()
    model_version = identity.replace(":", "_").replace(" ", "_")
    digest = hashlib.sha256(f"{identity}\0{text}".encode("utf-8")).hexdigest()
    return f"embed:{model_version}:{digest}"


async def get_text_embeddings_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """批量向量化（跳过 Redis，适合回填脚本）。"""
    if not texts:
        return []
    model = await asyncio.wait_for(
        asyncio.to_thread(get_embed_model), timeout=float(settings.EMBEDDING_TIMEOUT_SECONDS)
    )

    def _encode():
        arr = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return [row.tolist() for row in arr]

    return await asyncio.wait_for(
        asyncio.to_thread(_encode), timeout=float(settings.EMBEDDING_TIMEOUT_SECONDS)
    )
