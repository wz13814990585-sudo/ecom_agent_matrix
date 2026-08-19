"""确定性的领域请求解析器。"""

from ecom_agent_matrix.modules.parsers.goods import GoodsRequest, parse_goods_request
from ecom_agent_matrix.modules.parsers.social import SocialRequest, parse_social_request
from ecom_agent_matrix.modules.parsers.stock import StockRequest, parse_stock_request
from ecom_agent_matrix.modules.parsers.ad import AdRequest, ProfitInputs, parse_ad_request
from ecom_agent_matrix.modules.parsers.crm import CRMRequest, parse_crm_request
from ecom_agent_matrix.modules.parsers.report import ReportRequest, parse_report_request
from ecom_agent_matrix.modules.parsers.risk import RiskRequest, parse_risk_request
from ecom_agent_matrix.modules.parsers.competitor import (
    CompetitorRequest,
    parse_competitor_request,
)
from ecom_agent_matrix.modules.parsers.data_check import (
    DataCheckRequest,
    parse_data_check_request,
)

__all__ = [
    "AdRequest",
    "CRMRequest",
    "CompetitorRequest",
    "DataCheckRequest",
    "GoodsRequest",
    "ProfitInputs",
    "ReportRequest",
    "RiskRequest",
    "SocialRequest",
    "StockRequest",
    "parse_ad_request",
    "parse_competitor_request",
    "parse_crm_request",
    "parse_data_check_request",
    "parse_goods_request",
    "parse_report_request",
    "parse_risk_request",
    "parse_social_request",
    "parse_stock_request",
]
