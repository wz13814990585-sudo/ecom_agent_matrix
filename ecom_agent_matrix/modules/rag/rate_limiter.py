"""RAG 检索限流：控制并发，保护向量库与 rerank 模型。"""
import asyncio

from ecom_agent_matrix.config.settings import settings

_rag_semaphore: asyncio.Semaphore | None = None


def get_rag_semaphore() -> asyncio.Semaphore:
    global _rag_semaphore
    if _rag_semaphore is None:
        _rag_semaphore = asyncio.Semaphore(settings.RAG_MAX_CONCURRENT)
    return _rag_semaphore
