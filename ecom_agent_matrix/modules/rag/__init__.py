"""MCP-independent RAG contracts and service。"""

from .schemas import RAGAnswerResult, RAGDocument, RAGRequest, RAGRetrievalResult
from .service import RAGService, rag_service

__all__ = [
    "RAGAnswerResult",
    "RAGDocument",
    "RAGRequest",
    "RAGRetrievalResult",
    "RAGService",
    "rag_service",
]
