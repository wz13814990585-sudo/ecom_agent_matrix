"""工具注册中心与统一权限执行上下文。"""
# core/skill/skill_registry.py
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, Iterator, Type

from ecom_agent_matrix.config.constants import AGENT_EXEC, AGENT_QUERY
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult

# 全局工具容器：key=skill_name，value=工具类
skill_container: Dict[str, Type[BaseSkill]] = {}


@dataclass(frozen=True)
class SkillExecutionContext:
    """一次 Agent 调用链的权限主体。"""

    agent_id: str


_execution_context: ContextVar[SkillExecutionContext | None] = ContextVar(
    "skill_execution_context",
    default=None,
)


@contextmanager
def skill_execution_context(agent_id: str) -> Iterator[SkillExecutionContext]:
    """为当前异步调用链绑定 Agent 身份，嵌套 Skill 自动继承。"""
    context = SkillExecutionContext(agent_id=agent_id)
    token = _execution_context.set(context)
    try:
        yield context
    finally:
        _execution_context.reset(token)


def register_skill(skill_cls: Type[BaseSkill]) -> Type[BaseSkill]:
    """装饰器：自动将工具类注册到全局容器"""
    skill_container[skill_cls.skill_name] = skill_cls
    return skill_cls


async def exec_skill(skill_name: str, params: dict) -> SkillResult:
    """
    全局统一工具调度函数
    :param skill_name: 工具名称
    :param params: 执行参数
    :return: 标准化执行结果
    """
    # 判断工具是否存在
    if skill_name not in skill_container:
        return SkillResult(
            success=False,
            error_msg=f"不存在该工具：{skill_name}"
        )
    skill_cls = skill_container[skill_name]
    context = _execution_context.get()

    # Query 权限严格 fail-closed：只有显式 read_only=True 且 side_effect=False 才允许。
    if context and context.agent_id == AGENT_QUERY:
        if skill_cls.read_only is not True or skill_cls.side_effect is not False:
            return SkillResult(
                success=False,
                error_msg=f"data_query 无权执行非只读 Skill：{skill_name}",
            )

    # Exec 保留现有业务能力；具体允许调用哪些 Skill 仍由既有 workflow 决定。
    if context and context.agent_id not in {AGENT_QUERY, AGENT_EXEC}:
        return SkillResult(
            success=False,
            error_msg=f"未授权的 Skill execution context：{context.agent_id}",
        )

    skill_instance = skill_cls()
    result = await skill_instance.run(params)
    return result
