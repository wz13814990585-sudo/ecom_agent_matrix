# MCP消息优先级
MSG_PRIORITY_RISK = 0      # 风控预警（最高）
MSG_PRIORITY_CUSTOMER = 1  # 客服工单
MSG_PRIORITY_AD = 2        # 广告优化
MSG_PRIORITY_NORMAL = 3    # 日常上新、报表
MSG_PRIORITY_SOCIAL = 4    # 社媒生成（最低，大促可丢弃）

# 支持语种
LANG_LIST = ["en", "zh", "es", "fr"]

# Agent 唯一标识：按任务类型拆分（查询 / 执行 / 知识库 / 总调度），不按业务表拆分
AGENT_MASTER = "master_planning"
AGENT_QUERY = "data_query"
AGENT_EXEC = "biz_exec"
AGENT_RAG = "knowledge_rag"

# 历史实体 Agent id（不再注册为独立进程，仅供规划器别名 / 旧测试引用）
AGENT_GOODS = "goods_lookup"
AGENT_CRM = "customer_service"
AGENT_AD = "ad_optimizer"
AGENT_PRICE_WARN = "price_monitor_warn"
AGENT_STOCK = "stock_agent"
AGENT_SOCIAL = "social_marketing"
AGENT_DATA_CHECK = "data_check_agent"
AGENT_REPORT = "report_agent"

# 数据库表常量
TABLE_GOODS = "ecom_goods"
TABLE_ORDER = "ecom_order"
TABLE_COMPETITOR = "competitor_price"
TABLE_RISK_LOG = "risk_record"
TABLE_VECTOR_GOODS = "vector_goods_kb"
TABLE_FINETUNE_DATA = "finetune_dataset"