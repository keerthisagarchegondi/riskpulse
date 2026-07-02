-- =============================================================================
-- Migration 007: Create Feature Store Tables
-- RiskPulse Fraud Analytics Platform
-- =============================================================================
-- Stores computed ML features per transaction for the feature engineering
-- pipeline. Supports both real-time feature retrieval and historical
-- feature analysis.
-- =============================================================================

BEGIN;

-- Transaction Features Table (computed features per transaction)
CREATE TABLE transaction_features (
    feature_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id UUID NOT NULL REFERENCES transactions(transaction_id),
    customer_id VARCHAR(64) NOT NULL,

    -- Transaction features
    amount_zscore DECIMAL(10, 6) DEFAULT 0,
    hour_of_day SMALLINT NOT NULL DEFAULT 0 CHECK (hour_of_day >= 0 AND hour_of_day <= 23),
    day_of_week SMALLINT NOT NULL DEFAULT 0 CHECK (day_of_week >= 0 AND day_of_week <= 6),
    is_weekend BOOLEAN NOT NULL DEFAULT FALSE,
    is_holiday BOOLEAN NOT NULL DEFAULT FALSE,
    time_since_last_transaction DECIMAL(15, 2) DEFAULT 0,
    amount_to_avg_ratio DECIMAL(12, 6) DEFAULT 0,

    -- Velocity features
    txn_count_1h INTEGER NOT NULL DEFAULT 0,
    txn_count_24h INTEGER NOT NULL DEFAULT 0,
    txn_count_7d INTEGER NOT NULL DEFAULT 0,
    txn_amount_sum_1h DECIMAL(15, 2) NOT NULL DEFAULT 0,
    txn_amount_sum_24h DECIMAL(15, 2) NOT NULL DEFAULT 0,
    unique_merchants_24h INTEGER NOT NULL DEFAULT 0,
    unique_countries_24h INTEGER NOT NULL DEFAULT 0,

    -- Behavioral features
    new_merchant_flag BOOLEAN NOT NULL DEFAULT FALSE,
    unusual_hour_flag BOOLEAN NOT NULL DEFAULT FALSE,
    amount_percentile DECIMAL(7, 6) DEFAULT 0.5,
    channel_switch_flag BOOLEAN NOT NULL DEFAULT FALSE,

    -- Sequence features
    consecutive_declined_count INTEGER NOT NULL DEFAULT 0,
    rapid_succession_flag BOOLEAN NOT NULL DEFAULT FALSE,

    -- Metadata
    feature_version VARCHAR(20) NOT NULL DEFAULT '1.0',
    computation_latency_ms DECIMAL(8, 4),
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_transaction_features UNIQUE (transaction_id, feature_version)
);

CREATE INDEX idx_txn_features_transaction ON transaction_features(transaction_id);
CREATE INDEX idx_txn_features_customer ON transaction_features(customer_id);
CREATE INDEX idx_txn_features_computed_at ON transaction_features(computed_at);
CREATE INDEX idx_txn_features_customer_time ON transaction_features(customer_id, computed_at DESC);

-- Customer aggregation profiles for feature computation
CREATE TABLE customer_feature_profiles (
    customer_id VARCHAR(64) PRIMARY KEY,
    avg_transaction_amount DECIMAL(15, 2) DEFAULT 0,
    std_transaction_amount DECIMAL(15, 2) DEFAULT 0,
    min_transaction_amount DECIMAL(15, 2) DEFAULT 0,
    max_transaction_amount DECIMAL(15, 2) DEFAULT 0,
    total_transaction_count INTEGER DEFAULT 0,
    total_transaction_amount DECIMAL(18, 2) DEFAULT 0,
    last_transaction_timestamp TIMESTAMPTZ,
    last_channel VARCHAR(20),
    last_merchant_id VARCHAR(64),
    known_merchants JSONB DEFAULT '[]',
    known_countries JSONB DEFAULT '[]',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_customer_feature_profiles_updated ON customer_feature_profiles(updated_at);

-- Trigger for updated_at on customer_feature_profiles
CREATE TRIGGER trg_customer_feature_profiles_updated_at
    BEFORE UPDATE ON customer_feature_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Feature computation audit log
CREATE TABLE feature_computation_log (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id VARCHAR(64),
    transactions_processed INTEGER NOT NULL DEFAULT 0,
    features_computed INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    avg_latency_ms DECIMAL(8, 4),
    max_latency_ms DECIMAL(8, 4),
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_feature_computation_log_time ON feature_computation_log(completed_at DESC);

COMMIT;
