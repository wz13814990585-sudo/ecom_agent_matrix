"""RAG document normalization and citation-aware formatting。"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from ecom_agent_matrix.modules.rag.schemas import RAGCitation, RAGDocument

_SOURCE_KEYS = ("source_id", "document_id", "doc_id", "chunk_id")
_KNOWN_KEYS = {
    "sku", "goods_sku", "title", "product_name", "chunk_text", "content", "text",
    "chunk", "lang", "score", "vector_score", "bm25_score", "rrf_score",
    "relevance_score", "meta", "metadata", "citation_id", "source_id",
}


def _metadata(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("metadata", raw.get("meta", {}))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            value = {}
    base = dict(value) if isinstance(value, dict) else {}
    base.update({key: value for key, value in raw.items() if key not in _KNOWN_KEYS})
    return base


def stable_source_id(raw: dict[str, Any]) -> str:
    meta = _metadata(raw)
    for key in _SOURCE_KEYS:
        value = raw.get(key) or meta.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    sku = str(raw.get("sku") or raw.get("goods_sku") or "")
    text = str(
        raw.get("chunk_text")
        or raw.get("content")
        or raw.get("text")
        or raw.get("chunk")
        or ""
    )
    digest = hashlib.sha256(f"{sku}\0{text}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def normalize_documents(raw_documents: list[dict[str, Any]]) -> list[RAGDocument]:
    documents: list[RAGDocument] = []
    for raw in raw_documents:
        text = str(
            raw.get("chunk_text")
            or raw.get("content")
            or raw.get("text")
            or raw.get("chunk")
            or ""
        ).strip()
        if not text:
            continue
        meta = _metadata(raw)
        sku_value = raw.get("sku") or raw.get("goods_sku") or meta.get("sku")
        title = str(
            raw.get("title")
            or raw.get("product_name")
            or meta.get("title")
            or meta.get("product_name")
            or sku_value
            or ""
        )
        documents.append(
            RAGDocument(
                citation_id=f"S{len(documents) + 1}",
                source_id=stable_source_id(raw),
                sku=str(sku_value) if sku_value not in (None, "") else None,
                title=title,
                chunk_text=text,
                lang=raw.get("lang") or meta.get("lang"),
                score=raw.get("score"),
                vector_score=raw.get("vector_score"),
                bm25_score=raw.get("bm25_score"),
                rrf_score=raw.get("rrf_score"),
                relevance_score=raw.get("relevance_score"),
                metadata=meta,
            )
        )
    return documents


def citations_for_documents(documents: list[RAGDocument]) -> list[RAGCitation]:
    return [
        RAGCitation(
            citation_id=document.citation_id,
            source_id=document.source_id,
            title=document.title,
            sku=document.sku,
            snippet=document.chunk_text.replace("\n", " ")[:240],
        )
        for document in documents
    ]


def format_rag_context(documents: list[RAGDocument], limit: int = 5) -> str:
    lines: list[str] = []
    for document in documents[:limit]:
        label = document.title or document.sku or document.source_id
        lines.append(
            f"[{document.citation_id}] {label}: "
            f"{document.chunk_text.strip().replace(chr(10), ' ')[:500]}"
        )
    return "\n".join(lines)


def format_rag_docs(docs: list[dict[str, Any]] | list[RAGDocument], limit: int = 3) -> str:
    """旧 CRM formatting API 兼容层；新路径使用 citation-aware context。"""
    documents = (
        list(docs)
        if not docs or isinstance(docs[0], RAGDocument)
        else normalize_documents(docs)  # type: ignore[arg-type]
    )
    return format_rag_context(documents, limit=limit)  # type: ignore[arg-type]
