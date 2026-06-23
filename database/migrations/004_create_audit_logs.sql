-- =============================================================================
-- Migration 004: Create Audit Logs & Customer Profiles Tables
-- RiskPulse Fraud Analytics Platform
-- =============================================================================

BEGIN;

-- Customer Profiles (Velocity Features)
CREATE TABLE customer_profiles (
    customer_id VARCHAR(64) PRIMARY KEY,
    total_transactions_24h INTEGER NOT NULL DEFAULT 0,
    total_amount_24h DECIMAL(15, 2) NOT NULL DEFAULT 0,
    total_transactions_7d INTEGER NOT NULL DEFAULT 0,
    total_amount_7d DECIMAL(15, 2) NOT NULL DEFAULT 0,
    avg_transaction_amount DECIMAL(15, 2),
    max_transaction_amount DECIMAL(15, 2),
    unique_merchants_7d INTEGER NOT NULL DEFAULT 0,
    unique_countries_7d INTEGER NOT NULL DEFAULT 0,
    last_transaction_timestamp TIMESTAMPTZ,
    risk_tier VARCHAR(20) NOT NULL DEFAULT 'standard' CHECK (risk_tier IN ('low', 'standard', 'elevated', 'high')),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_customer_profiles_risk_tier ON customer_profiles(risk_tier);
CREATE INDEX idx_customer_profiles_updated ON customer_profiles(updated_at DESC);

CREATE TRIGGER trg_customer_profiles_updated_at
    BEFORE UPDATE ON customer_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Audit Logs
CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(128) NOT NULL,
    action VARCHAR(50) NOT NULL,
    actor VARCHAR(100),
    details JSONB DEFAULT '{}',
    ip_address INET,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_entity ON audit_logs(entity_type, entity_id);
CREATE INDEX idx_audit_logs_created ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_logs_actor ON audit_logs(actor) WHERE actor IS NOT NULL;
CREATE INDEX idx_audit_logs_event_type ON audit_logs(event_type);

-- Partitioning by month for audit logs (high volume)
-- Note: In production, consider partitioning this table by created_at

COMMENT ON TABLE customer_profiles IS 'Aggregated customer behavioral profiles for velocity-based fraud detection';
COMMENT ON COLUMN customer_profiles.risk_tier IS 'Customer risk classification: low, standard, elevated, high';
COMMENT ON TABLE audit_logs IS 'Immutable audit trail of all system actions';
COMMENT ON COLUMN audit_logs.actor IS 'User email or system service identifier (e.g., system:fraud_engine)';

COMMIT;
