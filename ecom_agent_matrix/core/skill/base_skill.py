"""工具抽象基类。"""
# core/skill/base_skill.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from ecom_agent_matrix.core.skill.spec import SkillSpec

# 所有工具统一返回结果模型
class SkillResult(BaseModel):
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: str = ""
    error_msg: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

class BaseSkill(ABC):
    """所有电商工具的父类，抽象约束，所有技能必须继承此类"""
    # 两个类属性必须子类重写
    skill_name: str    # 工具唯一标识，调用工具时使用
    skill_desc: str    # 工具功能描述，供LLM自动选择工具使用
    # 未显式声明的 Skill 一律按“可写、高风险”处理，保证 fail-closed。
    read_only: bool = False
    side_effect: bool = True
    risk_level: str = "high"
    skill_version: str = "1.0"
    timeout_seconds: float = 30.0
    idempotent: bool = False
    input_model: type[BaseModel] | None = None
    output_model: type[BaseModel] | None = None
    deprecated: bool = False
    replacement: str | None = None
    required_scopes: frozenset[str] = frozenset()
    approval_required: bool = False

    @classmethod
    def spec(cls) -> SkillSpec:
        """从兼容类属性生成统一契约，旧 Skill 无需一次性迁移。"""
        return SkillSpec(
            name=cls.skill_name,
            description=cls.skill_desc,
            version=cls.skill_version,
            read_only=cls.read_only,
            side_effect=cls.side_effect,
            risk_level=cls.risk_level,
            timeout_seconds=cls.timeout_seconds,
            idempotent=cls.idempotent,
            input_model=cls.input_model,
            output_model=cls.output_model,
            deprecated=cls.deprecated,
            replacement=cls.replacement,
            required_scopes=cls.required_scopes,
            approval_required=cls.approval_required,
        )

    @abstractmethod
    async def run(self, params: dict) -> SkillResult:
        """
        工具统一执行入口
        :param params: 工具执行所需全部参数，字典格式
        :return: 标准化SkillResult对象
        """
        pass
