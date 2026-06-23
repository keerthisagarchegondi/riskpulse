-- =============================================================================
-- Migration 003: Create Risk Scores Table
-- RiskPulse Fraud Analytics Platform
-- =============================================================================

BEGIN;

CREATE TABLE risk_scores (
    score_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id UUID NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    model_version VARCHAR(20) NOT NULL,
    overall_score DECIMAL(5, 4) NOT NULL CHECK (overall_score BETWEEN 0 AND 1),
    rule_score DECIMAL(5, 4) CHECK (rule_score BETWEEN 0 AND 1),
    anomaly_score DECIMAL(5, 4) CHECK (anomaly_score BETWEEN 0 AND 1),
    ml_score DECIMAL(5, 4) CHECK (ml_score BETWEEN 0 AND 1),
    feature_contributions JSONB DEFAULT '{}',
    scoring_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    latency_ms INTEGER CHECK (latency_ms >= 0)
);

-- Performance indexes
CREATE INDEX idx_risk_scores_transaction ON risk_scores(transaction_id);
CREATE INDEX idx_risk_scores_overall ON risk_scores(overall_score DESC);
CREATE INDEX idx_risk_scores_timestamp ON risk_scores(scoring_timestamp DESC);
CREATE INDEX idx_risk_scores_model ON risk_scores(model_version);

-- Composite for high-risk lookups
CREATE INDEX idx_risk_scores_high ON risk_scores(overall_score DESC, scoring_timestamp DESC)
    WHERE overall_score >= 0.8;

COMMENT ON TABLE risk_scores IS 'Detailed scoring breakdown per scored transaction';
COMMENT ON COLUMN risk_scores.overall_score IS 'Ensemble weighted score: 0.3*rule + 0.3*anomaly + 0.4*ml';
COMMENT ON COLUMN risk_scores.feature_contributions IS 'SHAP values for top contributing features';
COMMENT ON COLUMN risk_scores.latency_ms IS 'Total scoring computation time in milliseconds';

COMMIT;
