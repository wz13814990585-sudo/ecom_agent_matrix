"""Agent 注册中心。"""
# core/mcp/registry.py
import asyncio
from typing import Dict, Callable
from ecom_agent_matrix.core.mcp.bus import mcp_bus


# 全局Agent注册表：key=Agent标识  value=Agent异步执行函数
agent_map: Dict[str, Callable] = {}

def register_agent(agent_id: str):
    """装饰器：标记并注册任意Agent函数"""
    def decorator(agent_func: Callable):
        agent_map[agent_id] = agent_func
        return agent_func
    return decorator

async def start_all_agents():
    """批量启动全部已注册智能体，项目启动入口调用"""
    task_list = []
    for agent_id, agent_func in agent_map.items():
        # 给每个Agent分配专属订阅队列
        agent_msg_queue = mcp_bus.register_agent(agent_id)
        # 创建协程并发运行所有Agent
        task_list.append(asyncio.create_task(agent_func(agent_msg_queue)))
    # 等待所有Agent持续运行
    await asyncio.gather(*task_list)