"""Manual deterministic router benchmark; never calls a real LLM."""
from __future__ import annotations

import argparse
import statistics
import time

from ecom_agent_matrix.modules.agent_cluster.master_router import route_master_task

CASES = [
    {"query": "退款规则是什么"},
    {"query": "查询 SKU-BAG-001 库存", "sku": "SKU-BAG-001"},
    {"query": "帮我回复退款客户"},
]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100)
    args = parser.parse_args()
    latencies: list[float] = []
    successes = 0
    for index in range(max(1, args.n)):
        started = time.perf_counter()
        decision = route_master_task(CASES[index % len(CASES)])
        latencies.append((time.perf_counter() - started) * 1000)
        successes += int(bool(decision.target_agents or decision.mode == "clarify"))
    print({
        "benchmark": "deterministic_master_router",
        "runs": len(latencies),
        "p50_ms": round(statistics.median(latencies), 4),
        "p95_ms": round(percentile(latencies, 0.95), 4),
        "success_rate": round(successes / len(latencies), 4),
        "average_llm_calls": 0,
        "average_tokens": 0,
        "note": "No real LLM or external dependency was called.",
    })


if __name__ == "__main__":
    main()
