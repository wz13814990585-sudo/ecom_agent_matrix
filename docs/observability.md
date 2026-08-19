# Observability and resilience

The current runtime is intentionally a single-process asynchronous Agent system. This keeps the interview/demo deployment understandable and avoids distributed coordination costs. `MessageBus` remains an abstraction that could later use Redis Streams or RabbitMQ, but Phase 6 Lite does not implement distributed messaging.

`GET /metrics` exposes bounded-label Prometheus metrics. Production deployments should set `METRICS_AUTH_REQUIRED=true`; the caller then needs `system:read`.

Structured logs inherit `task_id` and hop-level `correlation_id` through `TraceContext`. Tenant and user identifiers are SHA-256 hashes; request bodies, queries, prompts, credentials and tokens are excluded or redacted.

Business POST routes use a process-local tenant/user rate limiter. This is suitable only for the single-process demo runtime. LLM and Taobao integrations use bounded component-level circuit breakers; existing LLM retry remains the sole retry layer and only handles transient failures.

Readiness checks PostgreSQL, Redis and Agent runtime. An unavailable/unconfigured LLM is reported as degraded and does not fail readiness unless `LLM_REQUIRED_FOR_READINESS=true`.

