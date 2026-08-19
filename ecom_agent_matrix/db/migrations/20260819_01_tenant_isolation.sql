BEGIN;

-- Add and backfill tenant ownership without deleting existing rows.
ALTER TABLE ecom_goods ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(64);
ALTER TABLE ecom_goods ADD COLUMN IF NOT EXISTS store_id VARCHAR(64);
ALTER TABLE ecom_order ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(64);
ALTER TABLE ecom_order ADD COLUMN IF NOT EXISTS store_id VARCHAR(64);
ALTER TABLE competitor_price ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(64);
ALTER TABLE competitor_price ADD COLUMN IF NOT EXISTS store_id VARCHAR(64);
ALTER TABLE risk_record ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(64);
ALTER TABLE risk_record ADD COLUMN IF NOT EXISTS store_id VARCHAR(64);
ALTER TABLE vector_goods_kb ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(64);
ALTER TABLE vector_goods_kb ADD COLUMN IF NOT EXISTS store_id VARCHAR(64);
ALTER TABLE agent_long_memory ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(64);
ALTER TABLE agent_long_memory ADD COLUMN IF NOT EXISTS store_id VARCHAR(64);
ALTER TABLE finetune_dataset ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(64);
ALTER TABLE finetune_dataset ADD COLUMN IF NOT EXISTS store_id VARCHAR(64);
ALTER TABLE mcp_message_log ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(64);
ALTER TABLE mcp_message_log ADD COLUMN IF NOT EXISTS store_id VARCHAR(64);

UPDATE ecom_goods SET tenant_id='demo_tenant', store_id=COALESCE(NULLIF(store_id,''),'demo_store') WHERE NULLIF(tenant_id,'') IS NULL OR NULLIF(store_id,'') IS NULL;
UPDATE ecom_order SET tenant_id='demo_tenant', store_id='demo_store' WHERE NULLIF(tenant_id,'') IS NULL OR NULLIF(store_id,'') IS NULL;
UPDATE competitor_price SET tenant_id='demo_tenant', store_id='demo_store' WHERE NULLIF(tenant_id,'') IS NULL OR NULLIF(store_id,'') IS NULL;
UPDATE risk_record SET tenant_id='demo_tenant', store_id='demo_store' WHERE NULLIF(tenant_id,'') IS NULL OR NULLIF(store_id,'') IS NULL;
UPDATE vector_goods_kb SET tenant_id='demo_tenant', store_id='demo_store' WHERE NULLIF(tenant_id,'') IS NULL OR NULLIF(store_id,'') IS NULL;
UPDATE agent_long_memory SET tenant_id='demo_tenant', store_id='demo_store' WHERE NULLIF(tenant_id,'') IS NULL OR NULLIF(store_id,'') IS NULL;
UPDATE finetune_dataset SET tenant_id='demo_tenant', store_id='demo_store' WHERE NULLIF(tenant_id,'') IS NULL OR NULLIF(store_id,'') IS NULL;
UPDATE mcp_message_log SET tenant_id='demo_tenant', store_id='demo_store' WHERE NULLIF(tenant_id,'') IS NULL OR NULLIF(store_id,'') IS NULL;

ALTER TABLE ecom_goods ALTER COLUMN tenant_id SET NOT NULL, ALTER COLUMN store_id SET NOT NULL;
ALTER TABLE ecom_order ALTER COLUMN tenant_id SET NOT NULL, ALTER COLUMN store_id SET NOT NULL;
ALTER TABLE competitor_price ALTER COLUMN tenant_id SET NOT NULL, ALTER COLUMN store_id SET NOT NULL;
ALTER TABLE risk_record ALTER COLUMN tenant_id SET NOT NULL, ALTER COLUMN store_id SET NOT NULL;
ALTER TABLE vector_goods_kb ALTER COLUMN tenant_id SET NOT NULL, ALTER COLUMN store_id SET NOT NULL;
ALTER TABLE agent_long_memory ALTER COLUMN tenant_id SET NOT NULL, ALTER COLUMN store_id SET NOT NULL;
ALTER TABLE finetune_dataset ALTER COLUMN tenant_id SET NOT NULL, ALTER COLUMN store_id SET NOT NULL;
ALTER TABLE mcp_message_log ALTER COLUMN tenant_id SET NOT NULL, ALTER COLUMN store_id SET NOT NULL;

ALTER TABLE ecom_goods DROP CONSTRAINT IF EXISTS ecom_goods_sku_key;
ALTER TABLE ecom_order DROP CONSTRAINT IF EXISTS ecom_order_order_no_key;
ALTER TABLE ecom_goods ADD CONSTRAINT ecom_goods_tenant_store_sku_key UNIQUE (tenant_id, store_id, sku);
ALTER TABLE ecom_order ADD CONSTRAINT ecom_order_tenant_store_order_key UNIQUE (tenant_id, store_id, order_no);
CREATE INDEX IF NOT EXISTS idx_goods_tenant_store_sku ON ecom_goods(tenant_id,store_id,sku);
CREATE INDEX IF NOT EXISTS idx_competitor_tenant_store_sku ON competitor_price(tenant_id,store_id,target_sku);
CREATE INDEX IF NOT EXISTS idx_risk_tenant_store_order ON risk_record(tenant_id,store_id,order_no);
CREATE INDEX IF NOT EXISTS idx_vector_goods_tenant_store_sku ON vector_goods_kb(tenant_id,store_id,goods_sku);
CREATE INDEX IF NOT EXISTS idx_agent_long_memory_scope_agent ON agent_long_memory(tenant_id,store_id,agent_name);

CREATE TABLE IF NOT EXISTS security_approval (
  approval_id UUID PRIMARY KEY, tenant_id VARCHAR(64) NOT NULL, store_id VARCHAR(64) NOT NULL,
  requester_user_id VARCHAR(64) NOT NULL, approver_user_id VARCHAR(64), task_id VARCHAR(64) NOT NULL,
  skill_name VARCHAR(128) NOT NULL, params_hash CHAR(64) NOT NULL,
  status VARCHAR(16) NOT NULL CHECK (status IN ('pending','approved','rejected','consumed','expired')),
  requested_at TIMESTAMPTZ NOT NULL, expires_at TIMESTAMPTZ NOT NULL,
  approved_at TIMESTAMPTZ, consumed_at TIMESTAMPTZ, reason_code VARCHAR(64)
);
CREATE TABLE IF NOT EXISTS security_audit_log (
  id BIGSERIAL PRIMARY KEY, event_type VARCHAR(64) NOT NULL, task_id VARCHAR(64),
  tenant_id VARCHAR(64) NOT NULL, store_id VARCHAR(64) NOT NULL, user_id VARCHAR(64),
  agent_id VARCHAR(64), skill_name VARCHAR(128), approval_id UUID, outcome VARCHAR(32),
  reason_code VARCHAR(64), metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  create_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_approval_scope_status ON security_approval(tenant_id,store_id,status,expires_at);
CREATE INDEX IF NOT EXISTS idx_security_audit_scope_time ON security_audit_log(tenant_id,store_id,create_time DESC);

-- RLS policy is identical for every tenant table and covers read and write.
DO $$
DECLARE table_name text;
BEGIN
  FOREACH table_name IN ARRAY ARRAY[
    'ecom_goods','ecom_order','competitor_price','risk_record','vector_goods_kb',
    'agent_long_memory','finetune_dataset','mcp_message_log','security_approval','security_audit_log'
  ] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
    EXECUTE format('DROP POLICY IF EXISTS tenant_store_isolation ON %I', table_name);
    EXECUTE format(
      'CREATE POLICY tenant_store_isolation ON %I USING (tenant_id = current_setting(''app.tenant_id'', true) AND store_id = current_setting(''app.store_id'', true)) WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true) AND store_id = current_setting(''app.store_id'', true))',
      table_name
    );
  END LOOP;
END $$;

COMMIT;
