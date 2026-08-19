"""工具抽象基类。"""
# core/skill/base_skill.py
from abc import ABC, abstractmethod
from pydantic import BaseModel

# 所有工具统一返回结果模型
class SkillResult(BaseModel):
    success: bool          # 是否执行成功
    data: dict = {}        # 成功时返回的业务数据
    error_msg: str = ""    # 失败时存储错误描述

class BaseSkill(ABC):
    """所有电商工具的父类，抽象约束，所有技能必须继承此类"""
    # 两个类属性必须子类重写
    skill_name: str    # 工具唯一标识，调用工具时使用
    skill_desc: str    # 工具功能描述，供LLM自动选择工具使用

    @abstractmethod
    async def run(self, params: dict) -> SkillResult:
        """
        工具统一执行入口
        :param params: 工具执行所需全部参数，字典格式
        :return: 标准化SkillResult对象
        """
        pass