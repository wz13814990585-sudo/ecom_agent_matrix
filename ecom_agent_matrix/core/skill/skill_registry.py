"""Skill 注册、查询与兼容调用入口。"""
# core/skill/skill_registry.py
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, Type

from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult
from ecom_agent_matrix.core.security import SecurityContext

if TYPE_CHECKING:
    from ecom_agent_matrix.core.tasking import TaskContext

# 全局工具容器：key=skill_name，value=工具类
skill_container: Dict[str, Type[BaseSkill]] = {}


@dataclass(frozen=True)
class SkillExecutionContext:
    """一次 Agent 调用链的权限主体。"""

    agent_id: str
    task_id: str = ""
    tenant_id: str = ""
    store_id: str = ""
    user_id: str = ""
    roles: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()
    identity_trusted: bool = False


_execution_context: ContextVar[SkillExecutionContext | None] = ContextVar(
    "skill_execution_context",
    default=None,
)


@contextmanager
def skill_execution_context(
    agent_id: str,
    *,
    task_context: "TaskContext | None" = None,
    security: SecurityContext | None = None,
    task_id: str = "",
) -> Iterator[SkillExecutionContext]:
    """为当前异步调用链绑定 Agent 身份，嵌套 Skill 自动继承。"""
    context = SkillExecutionContext(
        agent_id=agent_id,
        task_id=(task_context.task_id if task_context is not None else task_id),
        tenant_id=(security.tenant_id if security else (task_context.tenant_id or "") if task_context else ""),
        store_id=(security.store_id if security else (task_context.store_id or "") if task_context else ""),
        user_id=(security.user_id if security else (task_context.user_id or "") if task_context else ""),
        roles=security.roles if security else frozenset(),
        scopes=security.scopes if security else frozenset(),
        identity_trusted=bool(security and security.authenticated) or bool(
            task_context and task_context.identity_trusted
        ),
    )
    token = _execution_context.set(context)
    try:
        yield context
    finally:
        _execution_context.reset(token)


def current_skill_execution_context() -> SkillExecutionContext | None:
    """返回当前异步调用链继承的 Skill 身份。"""
    return _execution_context.get()


def register_skill(skill_cls: Type[BaseSkill]) -> Type[BaseSkill]:
    """注册 Skill，并在启动阶段校验契约与重复名称。"""
    if not isinstance(skill_cls, type) or not issubclass(skill_cls, BaseSkill):
        raise TypeError("register_skill 仅接受 BaseSkill 子类")

    spec = skill_cls.spec()
    name = spec.name.strip()
    if not name:
        raise ValueError("skill_name 不能为空")
    if name in skill_container:
        raise ValueError(f"Skill 重复注册：{name}")
    skill_container[name] = skill_cls
    return skill_cls


def lookup_skill(skill_name: str) -> Type[BaseSkill] | None:
    """按名称查询 Skill class。"""
    return skill_container.get(skill_name)


def list_skills() -> list[str]:
    """列出已注册 Skill 名称。"""
    return sorted(skill_container)


async def exec_skill(skill_name: str, params: dict) -> SkillResult:
    """向后兼容入口；实际执行统一委托给 SkillExecutor。"""
    from ecom_agent_matrix.core.skill.executor import skill_executor

    return await skill_executor.execute(skill_name, params)
