"""注册完整性：侧载 import 后应为 4 个任务类型 Agent。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.config.constants import (
    AGENT_EXEC,
    AGENT_MASTER,
    AGENT_QUERY,
    AGENT_RAG,
)
from ecom_agent_matrix.core.mcp.registry import agent_map
from ecom_agent_matrix.core.skill.skill_registry import skill_container

import ecom_agent_matrix.modules.agent_cluster  # noqa: F401
import ecom_agent_matrix.modules.skills  # noqa: F401


EXPECTED_AGENTS = {
    AGENT_MASTER,
    AGENT_QUERY,
    AGENT_EXEC,
    AGENT_RAG,
}

EXPECTED_SKILLS = {
    "profit_calc",
    "text_translate",
    "stock_predict",
    "order_risk_check",
    "social_media_gen",
    "ai_prompt_generate",
    "competitor_price",
    "price_monitor",
    "goods_sku_search",
    "goods_catalog",
    "ad_optimize",
    "data_integrity_check",
    "ops_report",
    "taobao_api",
    "crm_reply",
    "safe_sql_query",
}


def test_all_agents_registered():
    missing = EXPECTED_AGENTS - set(agent_map.keys())
    extra = set(agent_map.keys()) - EXPECTED_AGENTS
    assert not missing, f"未注册 Agent: {missing}"
    assert not extra, f"不应再注册实体 Agent: {extra}"
    assert len(agent_map) == 4


def test_all_skills_registered():
    missing = EXPECTED_SKILLS - set(skill_container.keys())
    assert not missing, f"未注册 Skill: {missing}"


if __name__ == "__main__":
    test_all_agents_registered()
    test_all_skills_registered()
    print(f"✅ agents={len(agent_map)} skills={len(skill_container)}")
