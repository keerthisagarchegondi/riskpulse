-- =============================================================================
-- Migration 001: Create Transactions Table
-- RiskPulse Fraud Analytics Platform
-- =============================================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE transactions (
    transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_transaction_id VARCHAR(64) NOT NULL UNIQUE,
    account_id VARCHAR(64) NOT NULL,
    customer_id VARCHAR(64) NOT NULL,
    merchant_id VARCHAR(64),
    merchant_name VARCHAR(255),
    merchant_category_code VARCHAR(10),
    transaction_amount DECIMAL(15, 2) NOT NULL CHECK (transaction_amount > 0),
    transaction_currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    transaction_type VARCHAR(20) NOT NULL CHECK (transaction_type IN ('purchase', 'withdrawal', 'transfer', 'refund')),
    channel VARCHAR(20) NOT NULL CHECK (channel IN ('online', 'pos', 'atm', 'mobile')),
    card_type VARCHAR(20) CHECK (card_type IN ('credit', 'debit', 'prepaid')),
    card_last_four VARCHAR(4) CHECK (card_last_four ~ '^\d{4}$'),
    ip_address INET,
    device_id VARCHAR(128),
    device_type VARCHAR(50),
    geo_latitude DECIMAL(10, 8) CHECK (geo_latitude BETWEEN -90 AND 90),
    geo_longitude DECIMAL(11, 8) CHECK (geo_longitude BETWEEN -180 AND 180),
    geo_country VARCHAR(3),
    geo_city VARCHAR(100),
    is_international BOOLEAN DEFAULT FALSE,
    transaction_timestamp TIMESTAMPTZ NOT NULL,
    processed_timestamp TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'declined', 'flagged')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX idx_transactions_account ON transactions(account_id);
CREATE INDEX idx_transactions_customer ON transactions(customer_id);
CREATE INDEX idx_transactions_timestamp ON transactions(transaction_timestamp DESC);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_transactions_merchant ON transactions(merchant_id) WHERE merchant_id IS NOT NULL;
CREATE INDEX idx_transactions_processed ON transactions(processed_timestamp DESC);

-- Composite index for common dashboard queries
CREATE INDEX idx_transactions_status_timestamp ON transactions(status, transaction_timestamp DESC);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_transactions_updated_at
    BEFORE UPDATE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

COMMENT ON TABLE transactions IS 'Core table storing all financial transaction events';
COMMENT ON COLUMN transactions.external_transaction_id IS 'External system reference ID (idempotency key)';
COMMENT ON COLUMN transactions.merchant_category_code IS 'ISO 18245 Merchant Category Code';
COMMENT ON COLUMN transactions.transaction_currency IS 'ISO 4217 currency code';
COMMENT ON COLUMN transactions.geo_country IS 'ISO 3166-1 alpha-3 country code';

COMMIT;
