"""Master Agent 单元测试：按任务类型规划（Query / Exec / RAG）。"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.config.constants import (
    AGENT_EXEC,
    AGENT_MASTER,
    AGENT_QUERY,
    AGENT_RAG,
)
from ecom_agent_matrix.core.mcp.message import MCPMessage
from ecom_agent_matrix.core.mcp.reply import build_reply
from ecom_agent_matrix.core.mcp.task_waiter import TaskReplyWaiter, is_agent_reply
from ecom_agent_matrix.modules.agent_cluster.master_agent import aggregate_sub_replies
from ecom_agent_matrix.modules.agent_cluster.master_planner import (
    AVAILABLE_AGENTS,
    ReactDecision,
    _guard_react_decision,
    _validate_agents,
    compute_memory_confidence,
    infer_task_type_from_query,
    merge_observation_into_working,
    plan_sub_tasks_crm_default,
    plan_sub_tasks_keyword,
    plan_sub_tasks_llm,
    plan_sub_tasks_rules,
    react_decide_rules,
    should_save_to_memory,
)


def test_is_agent_reply():
    req = MCPMessage(sender=AGENT_MASTER, target=AGENT_QUERY, priority=1, content={"task_type": "x"})
    rep = MCPMessage(
        sender=AGENT_QUERY,
        target=AGENT_MASTER,
        priority=1,
        content={"type": "agent_reply", "ref_task_id": "t1", "success": True, "data": {}},
    )
    assert not is_agent_reply(req)
    assert is_agent_reply(rep)


def test_plan_rules_injects_memory():
    plan = plan_sub_tasks_rules(
        {"task_type": "knowledge_qa", "query": "退款规则"},
        [{"content": "历史退款案例", "meta": {"confidence": 0.9}, "distance": 0.1}],
    )
    assert plan.sub_tasks[0]["target_agent"] == AGENT_RAG
    assert "_memory_context" in plan.sub_tasks[0]["payload"]


def test_keyword_infer_routes_refund_to_rag():
    assert infer_task_type_from_query("退款规则是什么") == "knowledge_qa"
    plan = plan_sub_tasks_keyword({"query": "连衣裙怎么搜索"}, [])
    agents = [s["target_agent"] for s in plan.sub_tasks]
    assert agents == [AGENT_QUERY]
    assert plan.planner == "keyword"


def test_stock_analysis_goes_to_query_only():
    """库存分析由 Query 内部解析 SKU，Master 不再拆 Goods→Stock。"""
    plan = plan_sub_tasks_rules({"task_type": "stock_analysis", "query": "防水户外背包备货"}, [])
    agents = [s["target_agent"] for s in plan.sub_tasks]
    assert agents == [AGENT_QUERY]

    plan2 = plan_sub_tasks_rules(
        {"task_type": "stock_analysis", "sku": "SKU-BAG-001", "query": "备货"},
        [],
    )
    assert [s["target_agent"] for s in plan2.sub_tasks] == [AGENT_QUERY]


def test_react_query_then_finish():
    working = {"query": "防水户外背包需要备货多少", "task_type": "stock_analysis"}
    d1 = react_decide_rules(working, [], [AGENT_QUERY])
    assert d1.action == "call_agent" and d1.agent == AGENT_QUERY

    obs = {
        "agent": AGENT_QUERY,
        "success": True,
        "data": {"query_kind": "stock", "sku": "SKU-BAG-001", "summary": "建议备货 40"},
        "error_msg": "",
    }
    working2 = merge_observation_into_working(working, obs)
    assert working2["sku"] == "SKU-BAG-001"
    d2 = react_decide_rules(working2, [obs], [AGENT_QUERY])
    assert d2.action == "finish"


def test_react_query_then_exec_when_both_suggested():
    working = {"query": "先查库存再生成报表"}
    d1 = react_decide_rules(working, [], [AGENT_QUERY, AGENT_EXEC])
    assert d1.agent == AGENT_QUERY
    obs = {"agent": AGENT_QUERY, "success": True, "data": {"query_kind": "stock"}, "error_msg": ""}
    d2 = react_decide_rules(working, [obs], [AGENT_QUERY, AGENT_EXEC])
    assert d2.action == "call_agent" and d2.agent == AGENT_EXEC


def test_bid_compare_is_query_not_ad():
    assert infer_task_type_from_query("我想知道防水背包的竞价对比") == "competitor_watch"
    assert infer_task_type_from_query("优化广告竞价 ROI") == "ad_optimize"
    plan = plan_sub_tasks_keyword({"query": "防水背包的竞价对比"}, [])
    assert plan.sub_tasks[0]["target_agent"] == AGENT_QUERY
    plan_ad = plan_sub_tasks_keyword({"query": "优化广告竞价 ROI"}, [])
    assert plan_ad.sub_tasks[0]["target_agent"] == AGENT_EXEC


def test_guard_maps_legacy_entity_agent():
    working = {"query": "防水背包的竞价对比"}
    bad = ReactDecision(
        thought="直接比价",
        action="call_agent",
        agent="price_monitor_warn",
        payload={"query": "防水背包的竞价对比"},
        source="llm",
    )
    fixed = _guard_react_decision(bad, working, [], [AGENT_QUERY])
    assert fixed.agent == AGENT_QUERY


def test_knowledge_qa_routes_to_rag():
    plan = plan_sub_tasks_rules({"task_type": "knowledge_qa", "query": "这款背包材质介绍一下"}, [])
    assert plan.sub_tasks[0]["target_agent"] == AGENT_RAG
    assert infer_task_type_from_query("商品知识问答：防水背包怎么用") == "knowledge_qa"
    assert infer_task_type_from_query("店铺退货政策是什么") == "knowledge_qa"


def test_only_three_sub_agents_registered_in_planner():
    assert set(AVAILABLE_AGENTS) == {AGENT_QUERY, AGENT_EXEC, AGENT_RAG}
    assert _validate_agents(["stock_predict", "ops_report", "data_integrity_check", "goods_rag"]) == [
        AGENT_QUERY,
        AGENT_EXEC,
        AGENT_RAG,
    ]


def test_social_product_and_platform_parse():
    from ecom_agent_matrix.modules.agent_cluster.handlers.social import (
        enrich_social_payload,
        extract_platform,
        extract_product_name,
    )

    assert extract_product_name({"query": "帮我生成tiktok文案，顺便对比价格"}) == ""
    assert extract_product_name({"query": "为「防水户外背包」生成tiktok文案"}) == "防水户外背包"
    plat, err = extract_platform({"platform": "xiaohongshu"})
    assert plat is None and "不支持" in err
    enriched = enrich_social_payload({"query": "为「防晒帽」写instagram文案"})
    assert enriched.get("product_name") == "防晒帽"
    assert enriched.get("platform") == "instagram"


def test_unknown_query_falls_back_to_rag():
    plan = plan_sub_tasks_keyword({"query": "随便说点什么xyz"}, [])
    assert plan.sub_tasks[0]["target_agent"] == AGENT_RAG
    assert plan.planner == "rag_default"
    assert plan.sub_tasks[0]["payload"].get("_fallback_route") is True


def test_rules_without_task_type_uses_keyword():
    plan = plan_sub_tasks_rules({"query": "帮我写社媒文案"}, [])
    assert plan.sub_tasks[0]["target_agent"] == AGENT_EXEC
    assert plan.planner == "keyword"


async def test_llm_fallback_to_keyword():
    with patch(
        "ecom_agent_matrix.modules.agent_cluster.master_planner.is_llm_configured",
        return_value=True,
    ), patch(
        "ecom_agent_matrix.modules.agent_cluster.master_planner.llm_chat",
        new=AsyncMock(side_effect=RuntimeError("llm down")),
    ):
        plan = await plan_sub_tasks_llm({"query": "库存不够了怎么办"}, [])
    assert plan.planner == "keyword_llm_fallback"
    agents = [s["target_agent"] for s in plan.sub_tasks]
    assert agents == [AGENT_QUERY]


async def test_llm_unknown_query_rag_default():
    with patch(
        "ecom_agent_matrix.modules.agent_cluster.master_planner.is_llm_configured",
        return_value=True,
    ), patch(
        "ecom_agent_matrix.modules.agent_cluster.master_planner.llm_chat",
        new=AsyncMock(side_effect=RuntimeError("llm down")),
    ):
        plan = await plan_sub_tasks_llm({"query": "??? unknown xyz"}, [])
    assert plan.sub_tasks[0]["target_agent"] == AGENT_RAG
    assert plan.planner == "keyword_llm_fallback"


def test_crm_default_alias_goes_rag():
    plan = plan_sub_tasks_crm_default({"query": ""}, [], reason="test")
    assert plan.sub_tasks[0]["target_agent"] == AGENT_RAG


async def test_llm_plan_parses_json():
    from ecom_agent_matrix.core.llm import ChatResult

    llm_json = json.dumps(
        {"agents": [AGENT_QUERY, AGENT_RAG], "confidence": 0.92, "reasoning": "先查数据再对照手册"}
    )
    with patch(
        "ecom_agent_matrix.modules.agent_cluster.master_planner.is_llm_configured",
        return_value=True,
    ), patch(
        "ecom_agent_matrix.modules.agent_cluster.master_planner.llm_chat",
        new=AsyncMock(return_value=ChatResult(content=llm_json)),
    ):
        plan = await plan_sub_tasks_llm({"query": "查订单退款规则"}, [])
    assert plan.planner == "llm"
    assert plan.plan_confidence == 0.92
    agents = [s["target_agent"] for s in plan.sub_tasks]
    assert AGENT_QUERY in agents and AGENT_RAG in agents


def test_memory_confidence_gate():
    final_ok = {"expected": 1, "received": 1, "timed_out": False, "all_success": True}
    final_bad = {"expected": 1, "received": 0, "timed_out": True, "all_success": False}
    ok, conf_ok = should_save_to_memory(0.9, final_ok)
    bad, _ = should_save_to_memory(0.9, final_bad)
    assert ok is True and conf_ok >= 0.75
    assert bad is False
    assert compute_memory_confidence(0.9, final_bad) < 0.75


async def test_wait_and_aggregate():
    task_id = "task-abc"
    TaskReplyWaiter.begin(task_id, 2)
    r1 = build_reply(
        MCPMessage(task_id=task_id, sender=AGENT_MASTER, target=AGENT_QUERY, priority=1, content={}),
        sender=AGENT_QUERY,
        success=True,
        data={"answer": "ok"},
    )
    TaskReplyWaiter.submit_reply(r1)
    replies = await TaskReplyWaiter.wait(task_id, timeout=1.0)
    agg = aggregate_sub_replies(task_id, replies, expected=2)
    assert agg["timed_out"] is True


def test_competitor_parse_helpers():
    from ecom_agent_matrix.modules.utils.competitor_parse import (
        extract_compete_price,
        extract_competitor,
    )

    assert extract_competitor({"query": "监控 AcmeShop 上 SKU-BAG-001"}) == "AcmeShop"
    assert extract_compete_price({"query": "价格29.99，运费5"}) == 29.99


if __name__ == "__main__":
    test_is_agent_reply()
    test_plan_rules_injects_memory()
    test_keyword_infer_routes_refund_to_rag()
    test_stock_analysis_goes_to_query_only()
    test_react_query_then_finish()
    test_react_query_then_exec_when_both_suggested()
    test_bid_compare_is_query_not_ad()
    test_guard_maps_legacy_entity_agent()
    test_knowledge_qa_routes_to_rag()
    test_only_three_sub_agents_registered_in_planner()
    test_social_product_and_platform_parse()
    test_unknown_query_falls_back_to_rag()
    test_rules_without_task_type_uses_keyword()
    test_crm_default_alias_goes_rag()
    test_memory_confidence_gate()
    test_competitor_parse_helpers()
    asyncio.run(test_llm_plan_parses_json())
    asyncio.run(test_llm_fallback_to_keyword())
    asyncio.run(test_llm_unknown_query_rag_default())
    asyncio.run(test_wait_and_aggregate())
    print("✅ 全部 Master/ReAct 测试通过")
