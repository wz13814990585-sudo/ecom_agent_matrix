"""Query / Exec 内部 handler：不是独立 Agent，不会注册进 agent_map。"""
from .ad import handle_ad
from .competitor import handle_price_warn
from .crm import handle_crm
from .data_check import handle_data_check, handle_risk
from .goods import handle_goods
from .report import handle_report
from .social import handle_social
from .stock import handle_stock

__all__ = [
    "handle_ad",
    "handle_crm",
    "handle_data_check",
    "handle_goods",
    "handle_price_warn",
    "handle_report",
    "handle_risk",
    "handle_social",
    "handle_stock",
]
