from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（.../跨境独立站电商自动化运营多智能体矩阵）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Postgres业务库配置（密码务必走 .env，勿提交真实口令）
    PG_HOST: str = "127.0.0.1"
    PG_PORT: int = 5432
    PG_USER: str = "postgres"
    PG_PWD: str = ""
    PG_DB: str = "ecom_matrix"

    # 模拟店铺标识（seed / 目录查询默认归属本店货盘）
    DEMO_STORE_ID: str = "demo_store"
    DEMO_STORE_NAME: str = "我的模拟独立站"

    # Redis缓存配置
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""  # 空=无密码（仅本地）；生产务必设置

    # 短记忆（CRM 会话）：滑动窗口 + Redis List 并发安全 append
    SHORT_MEMORY_TTL: int = 3600
    SHORT_MEMORY_MAX_MESSAGES: int = 20  # 只保留最近 N 条（约 10 轮对话）

    # 本地模型路径（预留；不存在时 embedding 自动回退 HuggingFace BAAI/bge-small-en-v1.5）
    LLM_MODEL_PATH: str = "./models/Qwen-7B"
    EMBED_MODEL_PATH: str = "./models/bge-small-en-v1.5"
    RERANK_MODEL_PATH: str = "./models/bge-reranker-base"

    # LLM 路由（架构层；具体供应商只是实现）
    LLM_PROVIDER: str = "deepseek"  # deepseek | openai
    LLM_DEFAULT_MODE: str = "chat"  # chat | reasoner
    LLM_TIMEOUT: float = 60.0
    LLM_MAX_RETRIES: int = 2  # 429/503/网络超时额外重试次数
    LLM_RETRY_BASE_DELAY: float = 0.8  # 指数退避基数（秒）

    # DeepSeek
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    # 兼容旧配置：未单独配置 CHAT_MODEL 时回退到 DEEPSEEK_MODEL
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_CHAT_MODEL: str = "deepseek-chat"
    DEEPSEEK_REASONER_MODEL: str = "deepseek-reasoner"
    DEEPSEEK_DEFAULT_MODE: str = "chat"  # chat | reasoner
    DEEPSEEK_TIMEOUT: float = 60.0
    DEEPSEEK_MAX_RETRIES: int = 2
    DEEPSEEK_RETRY_BASE_DELAY: float = 0.8
    DEEPSEEK_REASONER_MIN_TOKENS: int = 1024  # reasoner 的 max_tokens 下限（含思考链）

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_REASONER_MODEL: str = "o4-mini"
    OPENAI_TIMEOUT: float = 60.0
    OPENAI_REASONER_MIN_TOKENS: int = 1024

    # 淘宝开放平台 TOP（taobao_api skill；未配置则 skill 直接失败）
    TAOBAO_APP_KEY: str = ""
    TAOBAO_APP_SECRET: str = ""
    TAOBAO_SESSION_KEY: str = ""  # 用户授权 session；查订单等需登录态的接口必填
    TAOBAO_API_URL: str = "https://eco.taobao.com/router/rest"

    # MCP消息总线配置
    MCP_QUEUE_MAX_SIZE: int = 200
    MCP_TIMEOUT: int = 30
    MCP_RETRY_TIMES: int = 2  # 目标 Agent 尚未订阅时的额外重试次数

    # vLLM推理端口（预留）
    VLLM_PORT: int = 8001

    # RAG 检索配置
    RAG_MAX_CONCURRENT: int = 3
    RAG_CACHE_ENABLED: bool = True
    RAG_CACHE_TTL: int = 600
    RAG_INDEX_VERSION: str = "v1"
    RAG_RETRIEVAL_VERSION: str = "hybrid-v2"

    # 商品 SKU 检索：字面/trgm → 向量语义兜底
    GOODS_SEARCH_TRGM_MIN_SIM: float = 0.15
    GOODS_SEARCH_SEMANTIC_FALLBACK: bool = True
    GOODS_SEARCH_VECTOR_MAX_DIST: float = 0.65
    GOODS_SEARCH_VECTOR_TOP_K: int = 20

    # 竞品价格查询：demo=本地合成/库内缓存；http=自有价格适配器 API（COMPETITOR_PRICE_API_URL）
    COMPETITOR_PRICE_MODE: str = "demo"
    COMPETITOR_SPIDER_MODE: str = ""  # 已废弃，兼容旧 .env；请改用 COMPETITOR_PRICE_MODE
    COMPETITOR_PRICE_API_URL: str = ""
    COMPETITOR_PRICE_API_KEY: str = ""
    COMPETITOR_PRICE_API_TIMEOUT: float = 10.0
    COMPETITOR_HTTP_FALLBACK_DEMO: bool = True  # http 失败时是否回退 demo
    PRICE_WARN_MAX_CONCURRENT: int = 5

    # 社媒营销 Agent：process=进程内；redis=Redis 集合限流（不可用时回退 process）
    SOCIAL_MAX_CONCURRENT: int = 5
    SOCIAL_SKILL_TIMEOUT: float = 25.0
    SOCIAL_RATE_LIMIT_MODE: str = "process"

    # 广告优化 Agent
    AD_MAX_CONCURRENT: int = 5
    AD_SKILL_TIMEOUT: float = 25.0

    # 数据查询子 Agent（只读）
    QUERY_MAX_CONCURRENT: int = 5
    QUERY_SKILL_TIMEOUT: float = 60.0

    # 业务执行子 Agent（写操作）
    EXEC_MAX_CONCURRENT: int = 5
    EXEC_SKILL_TIMEOUT: float = 40.0

    # 数据校验（Query 内部 skill 超时）
    DATA_CHECK_MAX_CONCURRENT: int = 3
    DATA_CHECK_SKILL_TIMEOUT: float = 20.0

    # 报表（Exec 内部 skill 超时）
    REPORT_MAX_CONCURRENT: int = 3
    REPORT_SKILL_TIMEOUT: float = 30.0

    # Master Agent 配置
    MASTER_MAX_CONCURRENT: int = 8  # 用户级 Master 总任务并发上限
    MASTER_MAX_SUBTASK_CONCURRENT: int = 3  # ReAct 逐步下发时的并发上限
    MASTER_FAST_PATH_ENABLED: bool = True
    MASTER_MAX_PLAN_STEPS: int = 5
    MASTER_RECOVERY_MAX_STEPS: int = 2
    MASTER_MAX_LLM_CALLS: int = 3
    MASTER_REACT_MAX_STEPS: int = 5
    MASTER_PLAN_MODE: str = "reasoner"  # 初始规划：reasoner | chat
    MASTER_REACT_MODE: str = "chat"  # 逐步决策：默认 chat 降本；复杂任务可改 reasoner
    MASTER_PLAN_MAX_TOKENS: int = 2048
    MASTER_REACT_MAX_TOKENS: int = 800
    MASTER_REASONING_STORE_CHARS: int = 400  # 入库/回传截断思考链
    MASTER_PLAN_MIN_CONFIDENCE: float = 0.55
    MASTER_MEMORY_MIN_CONFIDENCE: float = 0.75
    MASTER_MEMORY_RECALL_MIN_CONFIDENCE: float = 0.6

    # HTTP API
    API_KEY: str = ""  # 空则本地不鉴权；生产请设置并走 X-API-Key
    API_REQUEST_TIMEOUT: float = 90.0
    API_SENDER: str = "api_gateway"

    # 最终结果 LLM 整理（API summary / Master 可读摘要）
    OUTPUT_POLISH_ENABLED: bool = True
    OUTPUT_POLISH_MAX_TOKENS: int = 400
    OUTPUT_POLISH_MAX_INPUT_CHARS: int = 3500

    # 子 Agent 解读层（RAG 答复 / 报表结构化摘要 / 竞品·库存·校验说明）
    AGENT_LLM_EXPLAIN_ENABLED: bool = True
    AGENT_LLM_EXPLAIN_MAX_TOKENS: int = 450


settings = Settings()
