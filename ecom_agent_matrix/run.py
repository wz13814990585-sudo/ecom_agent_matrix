"""项目统一启动入口：注册全部 Agent/Skill 后常驻运行。"""
from __future__ import annotations

import asyncio

# 侧载注册：@register_agent / @register_skill
import ecom_agent_matrix.modules.agent_cluster  # noqa: F401
import ecom_agent_matrix.modules.skills  # noqa: F401
from ecom_agent_matrix.core.logging_config import setup_logger
from ecom_agent_matrix.core.mcp.registry import agent_map, start_all_agents
from ecom_agent_matrix.core.skill.skill_registry import skill_container

logger = setup_logger("run")


async def main() -> None:
    logger.info(
        "matrix_starting",
        extra={
            "event": "matrix_starting",
            "agents": sorted(agent_map.keys()),
            "skills": sorted(skill_container.keys()),
        },
    )
    if not agent_map:
        raise RuntimeError("agent_map 为空：请确认 modules.agent_cluster 已正确 import")
    await start_all_agents()


if __name__ == "__main__":
    asyncio.run(main())
