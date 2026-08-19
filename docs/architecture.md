# Architecture sequences

The system has four runtime Agents. Workflows and Skills provide business variety without weakening those responsibility boundaries.

## Simple Fast Path

```mermaid
sequenceDiagram
    participant C as Client
    participant A as FastAPI
    participant M as Master
    participant Q as Query
    participant W as Read Workflow
    participant S as SkillExecutor
    participant D as PostgreSQL
    C->>A: authenticated simple query
    A->>M: MCPMessage(root task_id)
    M->>M: deterministic high-confidence route
    M->>Q: new correlation_id
    Q->>W: typed request
    W->>S: pure read Skill
    S->>D: tenant-scoped SELECT
    D-->>S: facts
    S-->>W: SkillResult
    W-->>Q: WorkflowResult
    Q-->>M: correlated reply
    M-->>A: final result, zero planner calls
    A-->>C: API response
```

## Composite Typed DAG

```mermaid
sequenceDiagram
    participant M as Master
    participant P as Typed Planner
    participant Q as Query
    participant R as RAG
    participant E as Exec/CRM
    M->>P: composite customer request
    P-->>M: validated DAG
    par independent context
        M->>Q: order_context
        Q-->>M: order facts
    and
        M->>R: policy_context
        R-->>M: grounded knowledge + citations
    end
    M->>E: customer_reply + upstream context
    E-->>M: final customer response
    Note over M: dependency-aware execution and bounded recovery
```

## Risk approval

```mermaid
sequenceDiagram
    participant U as Requester
    participant E as Exec Workflow
    participant X as SkillExecutor
    participant DB as PostgreSQL
    participant H as Human Approver
    U->>E: deterministic risky order payload
    E->>X: record_order_risk
    X->>DB: create exact-parameter pending approval
    X-->>U: APPROVAL_REQUIRED + approval_id
    H->>DB: authenticated approve endpoint
    DB-->>H: approved grant
    U->>E: same payload + X-Approval-Id
    E->>X: record_order_risk
    X->>DB: atomically consume approval
    X->>DB: write risk record once
    X-->>U: success
```

The LLM never participates in approval. Tenant identity, required scopes, parameter hashes, expiry, and one-time consumption are deterministic security controls.
