"""Agent 集群：只注册 4 个任务类型 Agent。

目录里真正会进 agent_map 的只有：
- master_agent.py   → master_planning
- query_agent.py    → data_query
- exec_agent.py     → biz_exec
- ../rag/rag_agent.py → knowledge_rag

handlers/ 下是内部业务函数，不是独立 Agent。
"""
from . import exec_agent, master_agent, query_agent
from ecom_agent_matrix.modules.rag import rag_agent  # noqa: F401

__all__ = [
    "exec_agent",
    "master_agent",
    "query_agent",
    "rag_agent",
]
