-- =============================================================================
-- Seed: Fraud Detection Rules
-- Initial production rules for the fraud detection engine
-- =============================================================================

INSERT INTO fraud_rules (rule_id, rule_name, rule_category, rule_expression, severity, is_active, threshold, description, created_by)
VALUES
-- Amount Rules
('RULE_HIGH_AMOUNT_01', 'High Amount - 3x Customer Average', 'amount',
 'transaction_amount > avg_transaction_amount * 3.0', 'high', TRUE, 3.0,
 'Flags transactions exceeding 3 times the customer average transaction amount',
 'system:seed'),

('RULE_HIGH_AMOUNT_02', 'Very High Amount - Absolute Threshold', 'amount',
 'transaction_amount > 10000', 'high', TRUE, 10000.0,
 'Flags single transactions exceeding $10,000 (BSA reporting threshold proximity)',
 'system:seed'),

('RULE_STRUCTURING_01', 'Possible Structuring - Just Below Threshold', 'amount',
 'transaction_amount BETWEEN 9000 AND 9999', 'medium', TRUE, 9000.0,
 'Detects amounts just below $10,000 reporting threshold (potential structuring)',
 'system:seed'),

('RULE_ROUND_AMOUNT_01', 'Suspicious Round Amount', 'amount',
 'transaction_amount % 100 = 0 AND transaction_amount >= 1000', 'low', TRUE, 1000.0,
 'Flags large round-number transactions which may indicate structuring',
 'system:seed'),

-- Velocity Rules
('RULE_VELOCITY_01', 'Rapid Successive Transactions', 'velocity',
 'txn_count_1h > 10', 'high', TRUE, 10.0,
 'More than 10 transactions within 1 hour for same account',
 'system:seed'),

('RULE_VELOCITY_02', 'High Daily Volume', 'velocity',
 'txn_count_24h > 50', 'medium', TRUE, 50.0,
 'More than 50 transactions in 24 hours',
 'system:seed'),

('RULE_VELOCITY_03', 'High Daily Amount', 'velocity',
 'total_amount_24h > 10000', 'high', TRUE, 10000.0,
 'Total amount exceeds $10,000 in 24-hour window',
 'system:seed'),

('RULE_VELOCITY_04', 'Rapid Succession Under 60 Seconds', 'velocity',
 'time_since_last_transaction < 60', 'critical', TRUE, 60.0,
 'Multiple transactions within 60 seconds (potential automated fraud)',
 'system:seed'),

-- Geographic Rules
('RULE_GEO_01', 'High-Risk Country Transaction', 'geo',
 'geo_country IN high_risk_countries', 'medium', TRUE, NULL,
 'Transaction originates from a high-risk country',
 'system:seed'),

('RULE_GEO_02', 'Impossible Travel', 'geo',
 'travel_speed_mph > 500', 'critical', TRUE, 500.0,
 'Sequential transactions from locations requiring travel speed > 500 mph',
 'system:seed'),

('RULE_GEO_03', 'International on Domestic Account', 'geo',
 'is_international = TRUE AND account_is_domestic_only = TRUE', 'high', TRUE, NULL,
 'International transaction on account flagged as domestic-only',
 'system:seed'),

-- Pattern Rules
('RULE_PATTERN_01', 'Multiple Declines Then Approve', 'pattern',
 'consecutive_declined_count >= 3 AND status = approved', 'high', TRUE, 3.0,
 'Approved transaction following 3+ consecutive declines (card testing pattern)',
 'system:seed'),

('RULE_PATTERN_02', 'New Device High Amount', 'pattern',
 'is_new_device = TRUE AND transaction_amount > avg_transaction_amount * 2.0', 'high', TRUE, 2.0,
 'High-value transaction from a device never seen before for this customer',
 'system:seed'),

('RULE_PATTERN_03', 'Channel Switch High Amount', 'pattern',
 'channel_switch_flag = TRUE AND transaction_amount > avg_transaction_amount * 2.0', 'medium', TRUE, 2.0,
 'High-value transaction from unusual channel for this customer',
 'system:seed'),

-- Temporal Rules
('RULE_TEMPORAL_01', 'Late Night High Value', 'temporal',
 'hour_of_day BETWEEN 1 AND 5 AND transaction_amount > 500', 'medium', TRUE, 500.0,
 'High-value transaction during unusual hours (1 AM - 5 AM local time)',
 'system:seed'),

('RULE_TEMPORAL_02', 'Weekend International', 'temporal',
 'is_weekend = TRUE AND is_international = TRUE AND transaction_amount > 1000', 'medium', TRUE, 1000.0,
 'High-value international transaction on weekend',
 'system:seed')

ON CONFLICT (rule_id) DO NOTHING;
