# Interview demo guide

These requests target the real FastAPI schemas. They assume the Docker/local quick start is complete, product vectors have been rebuilt, and API-key development mode is enabled with your own local value.

```bash
export BASE_URL=http://127.0.0.1:8000
export DEMO_API_KEY=your-local-demo-key
```

## 1. Simple Query Fast Path

```bash
curl -sS "$BASE_URL/api/v1/tasks" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $DEMO_API_KEY" \
  -d '{
    "query": "查询 SKU-BAG-001 商品信息",
    "task_type": "goods_search",
    "payload": {"sku": "SKU-BAG-001"}
  }'
```

Inspect `data.mode` (`fast_path`), `data.route`, and the single Query sub-result. Planner LLM calls should remain zero for this deterministic route.

## 2. Knowledge RAG

Run the embedding/indexing step before this request.

```bash
curl -sS "$BASE_URL/api/v1/customer/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $DEMO_API_KEY" \
  -d '{
    "query": "防水户外背包有什么特点？",
    "lang": "zh",
    "use_rag": true
  }'
```

The seed contains `SKU-BAG-001` with waterproof/lightweight product text. Inspect the RAG sub-result for vector and lexical candidate counts, RRF/rerank metadata, grounding status, and citations such as `[S1]`. Exact answer wording depends on whether an LLM provider is configured; source-context fallback remains available.

## 3. Composite DAG

```bash
curl -sS "$BASE_URL/api/v1/tasks" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $DEMO_API_KEY" \
  -d '{
    "query": "根据 ORD-20260301-001 的订单状态和退款规则帮我回复客户"
  }'
```

This exact phrasing matches the deterministic composite policy. The typed DAG creates parallel `order_context` (Query) and `policy_context` (RAG) steps, then passes both through `_upstream_context` to the dependent Exec/CRM step. The indexing script includes one explicitly marked demo refund-policy fixture so this RAG branch has a real matching source.

## 4. Deterministic risk approval

The amount `501` deterministically exceeds the risk threshold. First request approval:

```bash
RISK_RESPONSE=$(curl -sS "$BASE_URL/api/v1/tasks" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $DEMO_API_KEY" \
  -d '{
    "query": "检查高风险订单 ORD-DEMO-RISK",
    "task_type": "risk_control",
    "payload": {
      "order_no": "ORD-DEMO-RISK",
      "total_amount": 501,
      "buy_count": 1
    }
  }')
printf '%s\n' "$RISK_RESPONSE"
```

The response structure contains `APPROVAL_REQUIRED` and an approval ID under the first Fast Path sub-result. With `jq`, capture it as:

```bash
APPROVAL_ID=$(printf '%s' "$RISK_RESPONSE" | jq -r '.data.sub_results[0].data.approval_id')
```

Approve it with an identity that has `risk:approve` (the default demo `admin` role has this scope):

```bash
curl -sS -X POST "$BASE_URL/api/v1/approvals/$APPROVAL_ID/approve" \
  -H "X-API-Key: $DEMO_API_KEY"
```

Resubmit the exact same parameters and include the approval header:

```bash
curl -sS "$BASE_URL/api/v1/tasks" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $DEMO_API_KEY" \
  -H "X-Approval-Id: $APPROVAL_ID" \
  -d '{
    "query": "检查高风险订单 ORD-DEMO-RISK",
    "task_type": "risk_control",
    "payload": {
      "order_no": "ORD-DEMO-RISK",
      "total_amount": 501,
      "buy_count": 1
    }
  }'
```

Changing any approved Skill parameter invalidates the grant. A consumed approval cannot be reused, and the write Skill is never automatically retried.

## Smoke and benchmark commands

`smoke_e2e.py` returns structured failures for missing HTTP/dependency configuration instead of hiding them. `benchmark_demo.py` is limited to deterministic routing and does not claim end-to-end performance.
