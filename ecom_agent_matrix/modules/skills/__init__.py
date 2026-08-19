"""批量注册全部电商 Skill 模块。"""
from . import (
    ad_optimize,
    ai_prompt_gen,
    calc_tool,
    competitor_price,
    crm_reply,  # CRM：答复生成（RAG + LLM）
    data_integrity_check,
    goods_sku_search,
    goods_catalog,
    ops_report,
    price_monitor,
    risk_control,
    social_media,
    sql_tool,  # data_check：payload.sql / custom_sql
    stock_predict,
    taobao_api,  # CRM：use_taobao / taobao_method
    translate_tool,
)

__all__ = [
    "ad_optimize",
    "ai_prompt_gen",
    "calc_tool",
    "competitor_price",
    "crm_reply",
    "data_integrity_check",
    "goods_sku_search",
    "goods_catalog",
    "ops_report",
    "price_monitor",
    "risk_control",
    "social_media",
    "sql_tool",
    "stock_predict",
    "taobao_api",
    "translate_tool",
]
