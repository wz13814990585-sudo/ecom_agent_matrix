from __future__ import annotations

import math
import inspect

from ecom_agent_matrix.modules.rag import evaluation
from ecom_agent_matrix.modules.rag.evaluation import RAGEvalCase, evaluate_ranked_results


def test_eval_metrics_are_deterministic_and_correct():
    cases = [
        RAGEvalCase(query="q1", relevant_source_ids=["A", "B"]),
        RAGEvalCase(query="q2", relevant_source_ids=["C"]),
    ]
    ranked = {
        "q1": [{"source_id": "X"}, {"source_id": "A"}, {"source_id": "B"}],
        "q2": ["C", "Y"],
    }
    metrics = evaluate_ranked_results(cases, ranked, k=2)
    assert metrics.hit_rate_at_k == 1
    assert metrics.recall_at_k == 0.75
    assert metrics.mrr_at_k == 0.75
    q1_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert metrics.ndcg_at_k == (q1_ndcg + 1) / 2


def test_eval_empty_results_and_empty_cases_are_zero_and_do_not_need_llm():
    case = RAGEvalCase(query="q", relevant_source_ids=["A"])
    metrics = evaluate_ranked_results([case], {"q": []}, k=5)
    assert metrics.hit_rate_at_k == 0
    assert metrics.recall_at_k == 0
    assert metrics.mrr_at_k == 0
    assert metrics.ndcg_at_k == 0
    assert evaluate_ranked_results([], {}, k=5).case_count == 0
    source = inspect.getsource(evaluation)
    assert "rag_service.answer" not in source and "llm_explain" not in source
