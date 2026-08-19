"""MCP 消息数据模型。"""
# core/mcp/message.py
import uuid
from pydantic import BaseModel, Field
from ecom_agent_matrix.config.constants import MSG_PRIORITY_NORMAL
from ecom_agent_matrix.core.security import SecurityContext

class MCPMessage(BaseModel):
    """MCP全局统一消息数据模型，所有智能体通信强制使用该结构"""
    # 自动生成唯一任务ID，用于日志追溯、任务关联
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # 单条消息/子任务的请求-响应关联 ID；task_id 始终保留为用户请求根 ID。
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str          # 发送方Agent唯一标识（字符串常量，来自config.constants）
    target: str          # 接收方Agent唯一标识
    priority: int = MSG_PRIORITY_NORMAL  # 消息优先级，数字越小优先级越高
    content: dict        # 任务核心参数，字典存储任意业务数据（查询商品/竞品比价/生成文案等）
    security: SecurityContext | None = None
    create_time: float = Field(default_factory=lambda: uuid.uuid1().time)  # 消息创建时间戳
