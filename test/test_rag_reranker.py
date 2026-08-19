from __future__ import annotations

import asyncio
from unittest.mock import patch

from ecom_agent_matrix.modules.rag.formatter import normalize_documents
from ecom_agent_matrix.modules.rag.reranker import rerank_documents_detailed


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, *, show_progress_bar):
        self.calls.append((list(pairs), show_progress_bar))
        return self.scores


def test_batch_cross_encoder_predict_is_called_once_and_final_rank_gets_s1():
    model = FakeCrossEncoder([-3.0, 3.0, 0.0])
    candidates = [
        {"source_id": "A", "chunk_text": "unrelated", "rrf_score": 0.01},
        {"source_id": "B", "chunk_text": "refund policy", "rrf_score": 0.01},
        {"source_id": "C", "chunk_text": "refund", "rrf_score": 0.01},
    ]
    with patch(
        "ecom_agent_matrix.modules.rag.reranker._get_cross_encoder", return_value=model
    ):
        ranked, mode = asyncio.run(
            rerank_documents_detailed("refund policy", candidates, 2, threshold=-1)
        )
    assert mode == "cross_encoder"
    assert len(model.calls) == 1 and len(model.calls[0][0]) == 3
    assert len(ranked) == 2 and ranked[0]["source_id"] == "B"
    normalized = normalize_documents(ranked)
    assert normalized[0].source_id == "B" and normalized[0].citation_id == "S1"


def test_reranker_failure_uses_deterministic_keyword_fallback():
    candidates = [
        {"source_id": "A", "chunk_text": "refund policy", "rrf_score": 0.01},
        {"source_id": "B", "chunk_text": "shipping", "rrf_score": 0.02},
    ]
    with patch(
        "ecom_agent_matrix.modules.rag.reranker._get_cross_encoder",
        side_effect=FileNotFoundError("local model absent"),
    ) as loader:
        ranked, mode = asyncio.run(
            rerank_documents_detailed("refund policy", candidates, 1, threshold=-1)
        )
    assert mode == "keyword_fallback" and loader.call_count == 1
    assert len(ranked) == 1 and ranked[0]["source_id"] == "A"
    assert ranked[0]["semantic_score"] is None


def test_threshold_can_filter_every_candidate():
    with patch(
        "ecom_agent_matrix.modules.rag.reranker._get_cross_encoder",
        side_effect=FileNotFoundError,
    ):
        ranked, mode = asyncio.run(
            rerank_documents_detailed(
                "refund", [{"source_id": "A", "chunk_text": "shipping"}], 8, threshold=1
            )
        )
    assert ranked == [] and mode == "keyword_fallback"

