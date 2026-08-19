"""确定性的领域请求解析器。"""

from ecom_agent_matrix.modules.parsers.goods import GoodsRequest, parse_goods_request
from ecom_agent_matrix.modules.parsers.social import SocialRequest, parse_social_request
from ecom_agent_matrix.modules.parsers.stock import StockRequest, parse_stock_request
from ecom_agent_matrix.modules.parsers.competitor import (
    CompetitorRequest,
    parse_competitor_request,
)
from ecom_agent_matrix.modules.parsers.data_check import (
    DataCheckRequest,
    parse_data_check_request,
)

__all__ = [
    "GoodsRequest",
    "CompetitorRequest",
    "DataCheckRequest",
    "SocialRequest",
    "StockRequest",
    "parse_competitor_request",
    "parse_data_check_request",
    "parse_goods_request",
    "parse_social_request",
    "parse_stock_request",
]
