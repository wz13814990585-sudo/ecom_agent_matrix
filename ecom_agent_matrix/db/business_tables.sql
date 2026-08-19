-- 电商业务结构化数据表
-- 1.商品主表：存储多语种商品基础信息，RAG检索数据源（本店货盘）
CREATE TABLE IF NOT EXISTS ecom_goods (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'demo_tenant',
    store_id VARCHAR(64) NOT NULL DEFAULT 'demo_store',
    sku VARCHAR(64) NOT NULL,
    category VARCHAR(32),
    price DECIMAL(10,2),
    stock_num INT,
    title_en TEXT, title_zh TEXT, title_es TEXT, title_fr TEXT, -- 四国语言标题
    desc_multi TEXT,
    store_name VARCHAR(128) DEFAULT '我的模拟独立站',
    is_demo BOOLEAN DEFAULT true,                   -- true=演示数据，非真实上架
    create_time TIMESTAMP DEFAULT NOW(),
    update_time TIMESTAMP DEFAULT NOW()
    ,UNIQUE (tenant_id, store_id, sku)
);

-- 兼容已有库：补齐店铺标记字段
ALTER TABLE ecom_goods ADD COLUMN IF NOT EXISTS store_id VARCHAR(64) DEFAULT 'demo_store';
ALTER TABLE ecom_goods ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(64) DEFAULT 'demo_tenant';
ALTER TABLE ecom_goods ADD COLUMN IF NOT EXISTS store_name VARCHAR(128) DEFAULT '我的模拟独立站';
ALTER TABLE ecom_goods ADD COLUMN IF NOT EXISTS is_demo BOOLEAN DEFAULT true;

-- 2.订单表：库存预测Agent数据源
CREATE TABLE IF NOT EXISTS ecom_order (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'demo_tenant',
    store_id VARCHAR(64) NOT NULL DEFAULT 'demo_store',
    order_no VARCHAR(64),
    sku VARCHAR(64),
    buy_num INT,
    total_amount DECIMAL(10,2),
    refund_flag BOOLEAN DEFAULT false,
    create_time TIMESTAMP DEFAULT NOW(),
    UNIQUE (tenant_id, store_id, order_no)
);

-- 3.竞品监控表：竞品 Agent 写入价格快照（crawl_time 为入库时间字段名，兼容历史 schema）
CREATE TABLE IF NOT EXISTS competitor_price (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'demo_tenant',
    store_id VARCHAR(64) NOT NULL DEFAULT 'demo_store',
    target_sku VARCHAR(64),
    competitor_name VARCHAR(64),
    compete_price DECIMAL(10,2),
    crawl_time TIMESTAMP DEFAULT NOW()
);

-- 4.风控记录表：订单风控工具异常写入
CREATE TABLE IF NOT EXISTS risk_record (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'demo_tenant',
    store_id VARCHAR(64) NOT NULL DEFAULT 'demo_store',
    order_no VARCHAR(64),
    risk_type VARCHAR(32),
    risk_desc TEXT,
    create_time TIMESTAMP DEFAULT NOW()
);

-- 5.微调数据集表：自动采集运营数据入库，用于LoRA训练
CREATE TABLE IF NOT EXISTS finetune_dataset (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'demo_tenant',
    store_id VARCHAR(64) NOT NULL DEFAULT 'demo_store',
    task_type VARCHAR(32), -- tool_call / goods_text / chat / social
    input_text TEXT,
    output_text TEXT,
    lang VARCHAR(8),
    create_time TIMESTAMP DEFAULT NOW()
);

-- 6.MCP消息持久化日志：消息总线故障回溯
CREATE TABLE IF NOT EXISTS mcp_message_log (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'demo_tenant',
    store_id VARCHAR(64) NOT NULL DEFAULT 'demo_store',
    task_id VARCHAR(64),
    sender_agent VARCHAR(32),
    target_agent VARCHAR(32),
    priority INT,
    msg_content JSONB, -- JSONB存储灵活的消息参数
    create_time TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS security_approval (
    approval_id UUID PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    store_id VARCHAR(64) NOT NULL,
    requester_user_id VARCHAR(64) NOT NULL,
    approver_user_id VARCHAR(64),
    task_id VARCHAR(64) NOT NULL,
    skill_name VARCHAR(128) NOT NULL,
    params_hash CHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL CHECK (status IN ('pending','approved','rejected','consumed','expired')),
    requested_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    approved_at TIMESTAMPTZ,
    consumed_at TIMESTAMPTZ,
    reason_code VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS security_audit_log (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(64) NOT NULL,
    task_id VARCHAR(64),
    tenant_id VARCHAR(64) NOT NULL,
    store_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64),
    agent_id VARCHAR(64),
    skill_name VARCHAR(128),
    approval_id UUID,
    outcome VARCHAR(32),
    reason_code VARCHAR(64),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    create_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 索引优化，提升查询速度
CREATE INDEX IF NOT EXISTS idx_goods_tenant_store_sku ON ecom_goods(tenant_id, store_id, sku);
CREATE INDEX IF NOT EXISTS idx_order_time ON ecom_order(create_time);
CREATE INDEX IF NOT EXISTS idx_competitor_tenant_store_sku ON competitor_price(tenant_id, store_id, target_sku);
CREATE INDEX IF NOT EXISTS idx_risk_tenant_store_order ON risk_record(tenant_id, store_id, order_no);
CREATE INDEX IF NOT EXISTS idx_approval_scope_status ON security_approval(tenant_id, store_id, status, expires_at);
CREATE INDEX IF NOT EXISTS idx_security_audit_scope_time ON security_audit_log(tenant_id, store_id, create_time DESC);

-- pg_trgm：加速 ILIKE '%xxx%' / similarity，数据量大时必备
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_goods_title_zh_trgm ON ecom_goods USING gin (title_zh gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_goods_title_en_trgm ON ecom_goods USING gin (title_en gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_goods_title_es_trgm ON ecom_goods USING gin (title_es gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_goods_title_fr_trgm ON ecom_goods USING gin (title_fr gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_goods_sku_trgm ON ecom_goods USING gin (sku gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_goods_desc_trgm ON ecom_goods USING gin (desc_multi gin_trgm_ops);

-- ========== 示例种子数据（可重复执行） ==========

-- 商品主表
INSERT INTO ecom_goods (tenant_id, store_id, sku, category, price, stock_num, title_en, title_zh, title_es, title_fr, desc_multi)
VALUES
    ('demo_tenant', 'demo_store', 'SKU-BAG-001', 'bags', 49.99, 120,
     'Waterproof Outdoor Bag', '防水户外背包', 'Mochila impermeable', 'Sac outdoor imperméable',
     'Lightweight waterproof daypack for hiking and beach trips.'),
    ('demo_tenant', 'demo_store', 'SKU-DRESS-002', 'apparel', 39.90, 80,
     'Beach Summer Dress', '平价海边连衣裙', 'Vestido de playa', 'Robe de plage',
     'Breathable linen-blend dress for coastal vacation.'),
    ('demo_tenant', 'demo_store', 'SKU-SHOES-003', 'footwear', 59.00, 60,
     'Trail Running Shoes', '越野跑鞋', 'Zapatillas de trail', 'Chaussures de trail',
     'Grip sole trail shoes for wet and rocky paths.'),
    ('demo_tenant', 'demo_store', 'SKU-HAT-004', 'accessories', 19.50, 200,
     'UV Protection Hat', '防晒遮阳帽', 'Sombrero UV', 'Chapeau anti-UV',
     'Wide-brim sun hat with UPF50 protection.'),
    ('demo_tenant', 'demo_store', 'SKU-TENT-005', 'camping', 129.00, 35,
     '2-Person Camping Tent', '双人露营帐篷', 'Tienda para 2 personas', 'Tente 2 personnes',
     'Quick-setup waterproof tent for weekend camping.')
ON CONFLICT (tenant_id, store_id, sku) DO NOTHING;

-- 订单表
INSERT INTO ecom_order (tenant_id, store_id, order_no, sku, buy_num, total_amount, refund_flag)
VALUES
    ('demo_tenant', 'demo_store', 'ORD-20260301-001', 'SKU-BAG-001', 2, 99.98, false),
    ('demo_tenant', 'demo_store', 'ORD-20260301-002', 'SKU-DRESS-002', 1, 39.90, false),
    ('demo_tenant', 'demo_store', 'ORD-20260302-003', 'SKU-SHOES-003', 1, 59.00, false),
    ('demo_tenant', 'demo_store', 'ORD-20260302-004', 'SKU-HAT-004', 3, 58.50, false),
    ('demo_tenant', 'demo_store', 'ORD-20260303-005', 'SKU-TENT-005', 1, 129.00, true)
ON CONFLICT (tenant_id, store_id, order_no) DO NOTHING;

-- 竞品价格表
INSERT INTO competitor_price (target_sku, competitor_name, compete_price)
VALUES
    ('SKU-BAG-001', 'Amazon', 52.99),
    ('SKU-BAG-001', 'AliExpress', 45.50),
    ('SKU-DRESS-002', 'Shein', 35.90),
    ('SKU-SHOES-003', 'Decathlon', 54.90),
    ('SKU-TENT-005', 'REI', 139.00);

-- 风控记录表
INSERT INTO risk_record (order_no, risk_type, risk_desc)
VALUES
    ('ORD-20260303-005', 'refund_abuse', '短时间内重复退款申请'),
    ('ORD-20260301-001', 'address_mismatch', '收货地址与账单地址不一致'),
    ('ORD-20260302-004', 'high_velocity', '同一账号 1 小时内下单超过阈值阈值');

-- 微调数据集表
INSERT INTO finetune_dataset (task_type, input_text, output_text, lang)
VALUES
    ('goods_text', 'Write a short title for waterproof outdoor bag', 'Waterproof Outdoor Hiking Daypack', 'en'),
    ('chat', '这款海边连衣裙适合什么场景？', '适合沙滩度假、海边散步和轻度旅行穿着。', 'zh'),
    ('social', 'Generate IG caption for UV hat', 'Stay cool under the sun ☀️ UPF50 protection for every adventure.', 'en'),
    ('tool_call', '查询 SKU-BAG-001 库存', 'stock_num=120', 'zh');

-- MCP 消息日志示例
INSERT INTO mcp_message_log (task_id, sender_agent, target_agent, priority, msg_content)
VALUES
    ('seed-task-001', 'master_planning', 'goods_rag', 1, '{"query": "waterproof outdoor bag", "lang": "en"}'::jsonb),
    ('seed-task-002', 'master_planning', 'goods_rag', 1, '{"query": "平价海边连衣裙", "lang": "zh"}'::jsonb),
    ('seed-task-003', 'master_planning', 'customer_service', 1, '{"order_no": "ORD-20260303-005", "intent": "refund_status"}'::jsonb);

-- Seed data is installed first. Runtime access is then forced through tenant/store RLS.
DO $$
DECLARE table_name text;
BEGIN
  FOREACH table_name IN ARRAY ARRAY['ecom_goods','ecom_order','competitor_price','risk_record','finetune_dataset','mcp_message_log','security_approval','security_audit_log'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
    EXECUTE format('DROP POLICY IF EXISTS tenant_store_isolation ON %I', table_name);
    EXECUTE format('CREATE POLICY tenant_store_isolation ON %I USING (tenant_id = current_setting(''app.tenant_id'', true) AND store_id = current_setting(''app.store_id'', true)) WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true) AND store_id = current_setting(''app.store_id'', true))', table_name);
  END LOOP;
END $$;
