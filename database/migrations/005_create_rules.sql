-- =============================================================================
-- Migration 005: Create Fraud Rules & Model Registry Tables
-- RiskPulse Fraud Analytics Platform
-- =============================================================================

BEGIN;

-- Fraud Rules Table
CREATE TABLE fraud_rules (
    rule_id VARCHAR(50) PRIMARY KEY,
    rule_name VARCHAR(255) NOT NULL,
    rule_category VARCHAR(50) NOT NULL CHECK (rule_category IN ('amount', 'velocity', 'geo', 'pattern', 'temporal', 'device', 'merchant')),
    rule_expression TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    threshold DECIMAL(15, 4),
    description TEXT,
    created_by VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_fraud_rules_category ON fraud_rules(rule_category);
CREATE INDEX idx_fraud_rules_active ON fraud_rules(is_active) WHERE is_active = TRUE;

CREATE TRIGGER trg_fraud_rules_updated_at
    BEFORE UPDATE ON fraud_rules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Model Registry Table
CREATE TABLE model_registry (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL CHECK (model_type IN ('isolation_forest', 'xgboost', 'lightgbm', 'ensemble', 'logistic_regression')),
    status VARCHAR(20) NOT NULL DEFAULT 'staging' CHECK (status IN ('staging', 'production', 'retired', 'failed')),
    metrics JSONB DEFAULT '{}',
    parameters JSONB DEFAULT '{}',
    artifact_path VARCHAR(500),
    trained_at TIMESTAMPTZ,
    deployed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_model_name_version UNIQUE (model_name, model_version)
);

CREATE INDEX idx_model_registry_name ON model_registry(model_name);
CREATE INDEX idx_model_registry_status ON model_registry(status);
CREATE INDEX idx_model_registry_production ON model_registry(model_name, status) WHERE status = 'production';

-- Data Quality Metrics Table (for tracking quality scores over time)
CREATE TABLE data_quality_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id VARCHAR(50) NOT NULL,
    check_name VARCHAR(100) NOT NULL,
    check_category VARCHAR(50) NOT NULL,
    passed BOOLEAN NOT NULL,
    score DECIMAL(5, 2) CHECK (score BETWEEN 0 AND 100),
    details JSONB DEFAULT '{}',
    records_checked INTEGER NOT NULL DEFAULT 0,
    records_failed INTEGER NOT NULL DEFAULT 0,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_quality_metrics_batch ON data_quality_metrics(batch_id);
CREATE INDEX idx_quality_metrics_checked ON data_quality_metrics(checked_at DESC);
CREATE INDEX idx_quality_metrics_failed ON data_quality_metrics(passed) WHERE passed = FALSE;

-- Quarantine Table (for invalid records)
CREATE TABLE quarantine (
    quarantine_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_topic VARCHAR(100) NOT NULL,
    original_payload JSONB NOT NULL,
    failure_reason TEXT NOT NULL,
    failure_stage VARCHAR(50) NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'retrying', 'resolved', 'dead')),
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_quarantine_status ON quarantine(status);
CREATE INDEX idx_quarantine_created ON quarantine(created_at DESC);
CREATE INDEX idx_quarantine_stage ON quarantine(failure_stage);

COMMENT ON TABLE fraud_rules IS 'Configurable fraud detection rules with YAML-compatible expressions';
COMMENT ON TABLE model_registry IS 'ML model version tracking with metrics and deployment lifecycle';
COMMENT ON TABLE data_quality_metrics IS 'Historical data quality check results for trend monitoring';
COMMENT ON TABLE quarantine IS 'Invalid records captured for investigation and reprocessing';

COMMIT;
