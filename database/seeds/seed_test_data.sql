-- =============================================================================
-- Seed: Test Data
-- Sample data for development and testing
-- =============================================================================

-- Insert test customer profiles
INSERT INTO customer_profiles (customer_id, total_transactions_24h, total_amount_24h, total_transactions_7d, total_amount_7d, avg_transaction_amount, max_transaction_amount, unique_merchants_7d, unique_countries_7d, last_transaction_timestamp, risk_tier)
VALUES
('CUST-001', 3, 250.00, 15, 1200.00, 85.50, 350.00, 8, 1, NOW() - INTERVAL '2 hours', 'standard'),
('CUST-002', 8, 1500.00, 45, 8500.00, 188.88, 2000.00, 12, 2, NOW() - INTERVAL '30 minutes', 'elevated'),
('CUST-003', 1, 50.00, 5, 320.00, 64.00, 120.00, 4, 1, NOW() - INTERVAL '1 day', 'low'),
('CUST-004', 12, 5200.00, 60, 25000.00, 416.67, 5000.00, 20, 4, NOW() - INTERVAL '5 minutes', 'high'),
('CUST-005', 0, 0.00, 2, 150.00, 75.00, 100.00, 2, 1, NOW() - INTERVAL '3 days', 'standard')
ON CONFLICT (customer_id) DO NOTHING;

-- Insert test transactions
INSERT INTO transactions (external_transaction_id, account_id, customer_id, merchant_id, merchant_name, merchant_category_code, transaction_amount, transaction_currency, transaction_type, channel, card_type, card_last_four, ip_address, device_id, device_type, geo_latitude, geo_longitude, geo_country, geo_city, is_international, transaction_timestamp, status)
VALUES
-- Normal transactions
('TXN-TEST-001', 'ACC-001', 'CUST-001', 'MERCH-AMZN', 'Amazon.com', '5411', 45.99, 'USD', 'purchase', 'online', 'credit', '4532', '203.0.113.10', 'dev-001-ios', 'mobile_ios', 40.7128, -74.0060, 'USA', 'New York', FALSE, NOW() - INTERVAL '3 hours', 'approved'),

('TXN-TEST-002', 'ACC-001', 'CUST-001', 'MERCH-STAR', 'Starbucks', '5814', 6.50, 'USD', 'purchase', 'pos', 'debit', '4532', NULL, NULL, 40.7580, -73.9855, 'USA', 'New York', FALSE, NOW() - INTERVAL '2 hours', 'approved'),

('TXN-TEST-003', 'ACC-002', 'CUST-002', 'MERCH-UBER', 'Uber Technologies', '4121', 32.00, 'USD', 'purchase', 'mobile', 'credit', '8901', '198.51.100.50', 'dev-002-android', 'mobile_android', 37.7749, -122.4194, 'USA', 'San Francisco', FALSE, NOW() - INTERVAL '1 hour', 'approved'),

-- Suspicious transaction (high amount for customer)
('TXN-TEST-004', 'ACC-003', 'CUST-003', 'MERCH-ELEC', 'Best Electronics', '5732', 2500.00, 'USD', 'purchase', 'online', 'credit', '1234', '192.0.2.100', 'dev-003-new', 'desktop_windows', 51.5074, -0.1278, 'GBR', 'London', TRUE, NOW() - INTERVAL '30 minutes', 'flagged'),

-- High-velocity customer
('TXN-TEST-005', 'ACC-004', 'CUST-004', 'MERCH-WIRE', 'Wire Transfer Svc', '4829', 5000.00, 'USD', 'transfer', 'online', 'credit', '7890', '203.0.113.200', 'dev-004-desk', 'desktop_macos', 25.7617, -80.1918, 'USA', 'Miami', FALSE, NOW() - INTERVAL '10 minutes', 'flagged'),

-- ATM withdrawal
('TXN-TEST-006', 'ACC-005', 'CUST-005', NULL, 'ATM-NYC-1234', '6011', 200.00, 'USD', 'withdrawal', 'atm', 'debit', '5678', NULL, NULL, 40.7484, -73.9857, 'USA', 'New York', FALSE, NOW() - INTERVAL '5 minutes', 'approved')
ON CONFLICT (external_transaction_id) DO NOTHING;

-- Insert test fraud alerts
INSERT INTO fraud_alerts (transaction_id, alert_type, rule_id, risk_score, severity, status, description, details)
SELECT
    t.transaction_id,
    'rule_based',
    'RULE_HIGH_AMOUNT_01',
    0.8500,
    'high',
    'open',
    'Transaction amount $2,500.00 is 39x above customer average of $64.00',
    '{"triggered_rules": ["RULE_HIGH_AMOUNT_01", "RULE_GEO_03"], "amount_ratio": 39.06, "customer_avg": 64.00}'::jsonb
FROM transactions t
WHERE t.external_transaction_id = 'TXN-TEST-004'
ON CONFLICT DO NOTHING;

INSERT INTO fraud_alerts (transaction_id, alert_type, rule_id, risk_score, severity, status, description, details)
SELECT
    t.transaction_id,
    'ensemble',
    'RULE_VELOCITY_03',
    0.9200,
    'critical',
    'investigating',
    'Wire transfer of $5,000 from high-velocity account (12 txns in 24h, $5,200 total)',
    '{"triggered_rules": ["RULE_VELOCITY_03", "RULE_HIGH_AMOUNT_02"], "txn_count_24h": 12, "total_amount_24h": 5200.00}'::jsonb
FROM transactions t
WHERE t.external_transaction_id = 'TXN-TEST-005'
ON CONFLICT DO NOTHING;
