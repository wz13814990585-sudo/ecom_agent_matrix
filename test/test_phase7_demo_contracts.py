from pathlib import Path

from ecom_agent_matrix.api.schemas import CustomerChatRequest, TaskCreateRequest
from ecom_agent_matrix.scripts.init_db import split_sql_statements
from ecom_agent_matrix.scripts.reembed_vectors import _demo_knowledge_payloads
from ecom_agent_matrix.scripts.smoke_e2e import _build_payload


def test_init_db_keeps_postgres_do_blocks_intact():
    root = Path(__file__).resolve().parents[1]
    for name in ("business_tables.sql", "vector_tables.sql"):
        sql = (root / "ecom_agent_matrix" / "db" / name).read_text(encoding="utf-8")
        statements = split_sql_statements(sql)
        do_blocks = [statement for statement in statements if statement.lstrip().startswith("DO $$")]
        assert len(do_blocks) == 1
        assert "FOREACH" in do_blocks[0] and do_blocks[0].rstrip().endswith("END $$")


def test_documented_refund_policy_fixture_is_present_and_scoped():
    fixtures = _demo_knowledge_payloads()
    assert len(fixtures) == 1
    tenant_id, store_id, sku, lang, text, metadata = fixtures[0]
    assert (tenant_id, store_id, sku, lang) == (
        "demo_tenant", "demo_store", "KB-REFUND-POLICY", "zh"
    )
    assert "30 天" in text and "退款" in text
    assert metadata["demo"] is True and metadata["category"] == "store_policy"


def test_documented_demo_payloads_match_real_api_schemas_and_routes():
    TaskCreateRequest.model_validate({
        "query": "查询 SKU-BAG-001 商品信息",
        "task_type": "goods_search",
        "payload": {"sku": "SKU-BAG-001"},
    })
    CustomerChatRequest.model_validate({
        "query": "防水户外背包有什么特点？", "lang": "zh", "use_rag": True,
    })
    TaskCreateRequest.model_validate({
        "query": "根据 ORD-20260301-001 的订单状态和退款规则帮我回复客户",
    })
    TaskCreateRequest.model_validate({
        "query": "检查高风险订单 ORD-DEMO-RISK",
        "task_type": "risk_control",
        "payload": {"order_no": "ORD-DEMO-RISK", "total_amount": 501, "buy_count": 1},
    })
    assert _build_payload("fast-path")[1]["task_type"] == "goods_search"
    assert _build_payload("rag")[1]["use_rag"] is True
    assert "退款规则" in _build_payload("composite")[1]["query"]


def test_ci_and_docker_artifacts_keep_phase7_delivery_contracts():
    root = Path(__file__).resolve().parents[1]
    ci = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    dockerfile = (root / "ecom_agent_matrix/docker/Dockerfile").read_text(encoding="utf-8")
    compose = (root / "ecom_agent_matrix/docker/docker-compose.yml").read_text(encoding="utf-8")
    assert "python-version: \"3.11\"" in ci
    assert "--select E9,F63,F7,F82" in ci and "pytest -q" in ci
    assert "ECOM_DOWNLOAD_RERANKER: \"0\"" in ci
    assert "USER appuser" in dockerfile
    assert "127.0.0.1:8000/health" in dockerfile
    assert "/health/ready" not in dockerfile
    assert compose.count("condition: service_healthy") == 2
    assert "01-business.sql" in compose and "02-vector.sql" in compose
