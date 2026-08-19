"""工具注册中心。"""
# core/skill/skill_registry.py
from typing import Dict, Type
from ecom_agent_matrix.core.skill.base_skill import BaseSkill, SkillResult

# 全局工具容器：key=skill_name，value=工具类
skill_container: Dict[str, Type[BaseSkill]] = {}

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
    # 实例化工具并执行
    skill_instance = skill_container[skill_name]()
    result = await skill_instance.run(params)
    return result