"""确定性的领域请求解析器。"""

from ecom_agent_matrix.modules.parsers.goods import GoodsRequest, parse_goods_request
from ecom_agent_matrix.modules.parsers.social import SocialRequest, parse_social_request

__all__ = [
    "GoodsRequest",
    "SocialRequest",
    "parse_goods_request",
    "parse_social_request",
]
