-- =============================================================================
-- Snowflake Procedure: Load Transactions from RAW to STAGING
-- Called by Airflow snowflake_load DAG
-- =============================================================================

CREATE OR REPLACE PROCEDURE STAGING.LOAD_TRANSACTIONS(BATCH_ID VARCHAR)
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
BEGIN
    -- Parse VARIANT data from RAW into typed STAGING columns
    INSERT INTO STAGING.STG_TRANSACTIONS (
        TRANSACTION_ID,
        EXTERNAL_TRANSACTION_ID,
        ACCOUNT_ID,
        CUSTOMER_ID,
        MERCHANT_ID,
        MERCHANT_NAME,
        MERCHANT_CATEGORY_CODE,
        TRANSACTION_AMOUNT,
        TRANSACTION_CURRENCY,
        TRANSACTION_TYPE,
        CHANNEL,
        CARD_TYPE,
        CARD_LAST_FOUR,
        IP_ADDRESS,
        DEVICE_ID,
        DEVICE_TYPE,
        GEO_LATITUDE,
        GEO_LONGITUDE,
        GEO_COUNTRY,
        GEO_CITY,
        IS_INTERNATIONAL,
        TRANSACTION_TIMESTAMP,
        PROCESSED_TIMESTAMP,
        STATUS,
        BATCH_ID
    )
    SELECT
        RAW_DATA:transaction_id::VARCHAR,
        RAW_DATA:external_transaction_id::VARCHAR,
        RAW_DATA:account_id::VARCHAR,
        RAW_DATA:customer_id::VARCHAR,
        RAW_DATA:merchant_id::VARCHAR,
        RAW_DATA:merchant_name::VARCHAR,
        RAW_DATA:merchant_category_code::VARCHAR,
        RAW_DATA:transaction_amount::NUMBER(15, 2),
        COALESCE(RAW_DATA:transaction_currency::VARCHAR, 'USD'),
        RAW_DATA:transaction_type::VARCHAR,
        RAW_DATA:channel::VARCHAR,
        RAW_DATA:card_type::VARCHAR,
        RAW_DATA:card_last_four::VARCHAR,
        RAW_DATA:ip_address::VARCHAR,
        RAW_DATA:device_id::VARCHAR,
        RAW_DATA:device_type::VARCHAR,
        RAW_DATA:geo_latitude::NUMBER(10, 8),
        RAW_DATA:geo_longitude::NUMBER(11, 8),
        RAW_DATA:geo_country::VARCHAR,
        RAW_DATA:geo_city::VARCHAR,
        COALESCE(RAW_DATA:is_international::BOOLEAN, FALSE),
        RAW_DATA:transaction_timestamp::TIMESTAMPTZ,
        RAW_DATA:processed_timestamp::TIMESTAMPTZ,
        RAW_DATA:status::VARCHAR,
        :BATCH_ID
    FROM RAW.TRANSACTIONS
    WHERE BATCH_ID = :BATCH_ID
      AND RAW_DATA:transaction_id IS NOT NULL
      AND RAW_DATA:account_id IS NOT NULL
      AND RAW_DATA:transaction_amount IS NOT NULL;

    LET rows_loaded := SQLROWCOUNT;

    RETURN 'Successfully loaded ' || :rows_loaded || ' transactions for batch ' || :BATCH_ID;
END;
$$;

-- =============================================================================
-- Procedure: Load Fraud Alerts from RAW to STAGING
-- =============================================================================

CREATE OR REPLACE PROCEDURE STAGING.LOAD_FRAUD_ALERTS(BATCH_ID VARCHAR)
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
BEGIN
    INSERT INTO STAGING.STG_FRAUD_ALERTS (
        ALERT_ID,
        TRANSACTION_ID,
        ALERT_TYPE,
        RULE_ID,
        RISK_SCORE,
        SEVERITY,
        STATUS,
        DESCRIPTION,
        DETAILS,
        ASSIGNED_TO,
        RESOLVED_AT,
        RESOLUTION_NOTES,
        CREATED_AT,
        BATCH_ID
    )
    SELECT
        RAW_DATA:alert_id::VARCHAR,
        RAW_DATA:transaction_id::VARCHAR,
        RAW_DATA:alert_type::VARCHAR,
        RAW_DATA:rule_id::VARCHAR,
        RAW_DATA:risk_score::NUMBER(5, 4),
        RAW_DATA:severity::VARCHAR,
        COALESCE(RAW_DATA:status::VARCHAR, 'open'),
        RAW_DATA:description::VARCHAR,
        RAW_DATA:details::VARIANT,
        RAW_DATA:assigned_to::VARCHAR,
        RAW_DATA:resolved_at::TIMESTAMPTZ,
        RAW_DATA:resolution_notes::VARCHAR,
        RAW_DATA:created_at::TIMESTAMPTZ,
        :BATCH_ID
    FROM RAW.FRAUD_ALERTS
    WHERE BATCH_ID = :BATCH_ID
      AND RAW_DATA:alert_id IS NOT NULL
      AND RAW_DATA:transaction_id IS NOT NULL;

    LET rows_loaded := SQLROWCOUNT;

    RETURN 'Successfully loaded ' || :rows_loaded || ' fraud alerts for batch ' || :BATCH_ID;
END;
$$;

-- =============================================================================
-- Procedure: Load Risk Scores from RAW to STAGING
-- =============================================================================

CREATE OR REPLACE PROCEDURE STAGING.LOAD_RISK_SCORES(BATCH_ID VARCHAR)
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
BEGIN
    INSERT INTO STAGING.STG_RISK_SCORES (
        SCORE_ID,
        TRANSACTION_ID,
        MODEL_VERSION,
        OVERALL_SCORE,
        RULE_SCORE,
        ANOMALY_SCORE,
        ML_SCORE,
        FEATURE_CONTRIBUTIONS,
        SCORING_TIMESTAMP,
        LATENCY_MS,
        BATCH_ID
    )
    SELECT
        RAW_DATA:score_id::VARCHAR,
        RAW_DATA:transaction_id::VARCHAR,
        RAW_DATA:model_version::VARCHAR,
        RAW_DATA:overall_score::NUMBER(5, 4),
        RAW_DATA:rule_score::NUMBER(5, 4),
        RAW_DATA:anomaly_score::NUMBER(5, 4),
        RAW_DATA:ml_score::NUMBER(5, 4),
        RAW_DATA:feature_contributions::VARIANT,
        RAW_DATA:scoring_timestamp::TIMESTAMPTZ,
        RAW_DATA:latency_ms::NUMBER,
        :BATCH_ID
    FROM RAW.RISK_SCORES
    WHERE BATCH_ID = :BATCH_ID
      AND RAW_DATA:score_id IS NOT NULL
      AND RAW_DATA:transaction_id IS NOT NULL;

    LET rows_loaded := SQLROWCOUNT;

    RETURN 'Successfully loaded ' || :rows_loaded || ' risk scores for batch ' || :BATCH_ID;
END;
$$;

-- =============================================================================
-- Procedure: Load Customer Profiles from RAW to STAGING
-- =============================================================================

CREATE OR REPLACE PROCEDURE STAGING.LOAD_CUSTOMER_PROFILES(BATCH_ID VARCHAR)
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
BEGIN
    MERGE INTO STAGING.STG_CUSTOMER_PROFILES tgt
    USING (
        SELECT
            RAW_DATA:customer_id::VARCHAR AS CUSTOMER_ID,
            RAW_DATA:total_transactions_24h::NUMBER AS TOTAL_TRANSACTIONS_24H,
            RAW_DATA:total_amount_24h::NUMBER(15, 2) AS TOTAL_AMOUNT_24H,
            RAW_DATA:total_transactions_7d::NUMBER AS TOTAL_TRANSACTIONS_7D,
            RAW_DATA:total_amount_7d::NUMBER(15, 2) AS TOTAL_AMOUNT_7D,
            RAW_DATA:avg_transaction_amount::NUMBER(15, 2) AS AVG_TRANSACTION_AMOUNT,
            RAW_DATA:max_transaction_amount::NUMBER(15, 2) AS MAX_TRANSACTION_AMOUNT,
            RAW_DATA:unique_merchants_7d::NUMBER AS UNIQUE_MERCHANTS_7D,
            RAW_DATA:unique_countries_7d::NUMBER AS UNIQUE_COUNTRIES_7D,
            RAW_DATA:last_transaction_timestamp::TIMESTAMPTZ AS LAST_TRANSACTION_TIMESTAMP,
            RAW_DATA:risk_tier::VARCHAR AS RISK_TIER,
            :BATCH_ID AS BATCH_ID
        FROM RAW.TRANSACTIONS
        WHERE BATCH_ID = :BATCH_ID
          AND RAW_DATA:customer_id IS NOT NULL
    ) src
    ON tgt.CUSTOMER_ID = src.CUSTOMER_ID
    WHEN MATCHED THEN UPDATE SET
        tgt.TOTAL_TRANSACTIONS_24H = src.TOTAL_TRANSACTIONS_24H,
        tgt.TOTAL_AMOUNT_24H = src.TOTAL_AMOUNT_24H,
        tgt.TOTAL_TRANSACTIONS_7D = src.TOTAL_TRANSACTIONS_7D,
        tgt.TOTAL_AMOUNT_7D = src.TOTAL_AMOUNT_7D,
        tgt.AVG_TRANSACTION_AMOUNT = src.AVG_TRANSACTION_AMOUNT,
        tgt.MAX_TRANSACTION_AMOUNT = src.MAX_TRANSACTION_AMOUNT,
        tgt.UNIQUE_MERCHANTS_7D = src.UNIQUE_MERCHANTS_7D,
        tgt.UNIQUE_COUNTRIES_7D = src.UNIQUE_COUNTRIES_7D,
        tgt.LAST_TRANSACTION_TIMESTAMP = src.LAST_TRANSACTION_TIMESTAMP,
        tgt.RISK_TIER = src.RISK_TIER,
        tgt.BATCH_ID = src.BATCH_ID,
        tgt.LOADED_AT = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN INSERT (
        CUSTOMER_ID, TOTAL_TRANSACTIONS_24H, TOTAL_AMOUNT_24H,
        TOTAL_TRANSACTIONS_7D, TOTAL_AMOUNT_7D, AVG_TRANSACTION_AMOUNT,
        MAX_TRANSACTION_AMOUNT, UNIQUE_MERCHANTS_7D, UNIQUE_COUNTRIES_7D,
        LAST_TRANSACTION_TIMESTAMP, RISK_TIER, BATCH_ID
    ) VALUES (
        src.CUSTOMER_ID, src.TOTAL_TRANSACTIONS_24H, src.TOTAL_AMOUNT_24H,
        src.TOTAL_TRANSACTIONS_7D, src.TOTAL_AMOUNT_7D, src.AVG_TRANSACTION_AMOUNT,
        src.MAX_TRANSACTION_AMOUNT, src.UNIQUE_MERCHANTS_7D, src.UNIQUE_COUNTRIES_7D,
        src.LAST_TRANSACTION_TIMESTAMP, src.RISK_TIER, src.BATCH_ID
    );

    LET rows_loaded := SQLROWCOUNT;

    RETURN 'Successfully loaded ' || :rows_loaded || ' customer profiles for batch ' || :BATCH_ID;
END;
$$;

-- =============================================================================
-- Procedure: Transform STAGING to ANALYTICS fact table
-- =============================================================================

CREATE OR REPLACE PROCEDURE ANALYTICS.LOAD_FACT_TRANSACTIONS(BATCH_ID VARCHAR)
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
BEGIN
    INSERT INTO ANALYTICS.FACT_TRANSACTIONS (
        TRANSACTION_ID,
        DATE_KEY,
        CUSTOMER_KEY,
        MERCHANT_KEY,
        GEOGRAPHY_KEY,
        CHANNEL_KEY,
        TRANSACTION_AMOUNT,
        TRANSACTION_CURRENCY,
        RISK_SCORE,
        IS_FRAUD,
        IS_FLAGGED,
        ALERT_SEVERITY,
        PROCESSING_LATENCY_MS,
        TRANSACTION_TIMESTAMP
    )
    SELECT
        t.TRANSACTION_ID,
        TO_NUMBER(TO_CHAR(t.TRANSACTION_TIMESTAMP, 'YYYYMMDD')) AS DATE_KEY,
        c.CUSTOMER_KEY,
        m.MERCHANT_KEY,
        g.GEOGRAPHY_KEY,
        ch.CHANNEL_KEY,
        t.TRANSACTION_AMOUNT,
        t.TRANSACTION_CURRENCY,
        rs.OVERALL_SCORE,
        COALESCE(fa.STATUS = 'resolved' AND fa.STATUS != 'false_positive', FALSE) AS IS_FRAUD,
        COALESCE(t.STATUS = 'flagged', FALSE) AS IS_FLAGGED,
        fa.SEVERITY,
        rs.LATENCY_MS,
        t.TRANSACTION_TIMESTAMP
    FROM STAGING.STG_TRANSACTIONS t
    LEFT JOIN ANALYTICS.DIM_CUSTOMER c
        ON t.CUSTOMER_ID = c.CUSTOMER_ID AND c.IS_CURRENT = TRUE
    LEFT JOIN ANALYTICS.DIM_MERCHANT m
        ON t.MERCHANT_ID = m.MERCHANT_ID
    LEFT JOIN ANALYTICS.DIM_GEOGRAPHY g
        ON t.GEO_COUNTRY = g.COUNTRY_CODE AND COALESCE(t.GEO_CITY, '') = COALESCE(g.CITY, '')
    LEFT JOIN ANALYTICS.DIM_CHANNEL ch
        ON t.CHANNEL = ch.CHANNEL_CODE
    LEFT JOIN STAGING.STG_RISK_SCORES rs
        ON t.TRANSACTION_ID = rs.TRANSACTION_ID
    LEFT JOIN STAGING.STG_FRAUD_ALERTS fa
        ON t.TRANSACTION_ID = fa.TRANSACTION_ID
    WHERE t.BATCH_ID = :BATCH_ID;

    LET rows_loaded := SQLROWCOUNT;

    RETURN 'Loaded ' || :rows_loaded || ' rows into FACT_TRANSACTIONS for batch ' || :BATCH_ID;
END;
$$;

-- =============================================================================
-- Procedure: Update Customer Dimension (SCD Type 2)
-- =============================================================================

CREATE OR REPLACE PROCEDURE ANALYTICS.UPDATE_DIM_CUSTOMER(BATCH_ID VARCHAR)
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
BEGIN
    -- Close existing current records that have changed
    UPDATE ANALYTICS.DIM_CUSTOMER dc
    SET
        EFFECTIVE_TO = CURRENT_TIMESTAMP(),
        IS_CURRENT = FALSE
    FROM STAGING.STG_CUSTOMER_PROFILES sp
    WHERE dc.CUSTOMER_ID = sp.CUSTOMER_ID
      AND dc.IS_CURRENT = TRUE
      AND sp.BATCH_ID = :BATCH_ID
      AND (dc.RISK_TIER != sp.RISK_TIER
           OR dc.AVG_TRANSACTION_AMOUNT != sp.AVG_TRANSACTION_AMOUNT);

    -- Insert new current records
    INSERT INTO ANALYTICS.DIM_CUSTOMER (
        CUSTOMER_ID,
        RISK_TIER,
        TOTAL_LIFETIME_TRANSACTIONS,
        AVG_TRANSACTION_AMOUNT,
        MAX_TRANSACTION_AMOUNT,
        EFFECTIVE_FROM,
        EFFECTIVE_TO,
        IS_CURRENT
    )
    SELECT
        sp.CUSTOMER_ID,
        sp.RISK_TIER,
        sp.TOTAL_TRANSACTIONS_7D,
        sp.AVG_TRANSACTION_AMOUNT,
        sp.MAX_TRANSACTION_AMOUNT,
        CURRENT_TIMESTAMP(),
        '9999-12-31'::TIMESTAMPTZ,
        TRUE
    FROM STAGING.STG_CUSTOMER_PROFILES sp
    LEFT JOIN ANALYTICS.DIM_CUSTOMER dc
        ON sp.CUSTOMER_ID = dc.CUSTOMER_ID AND dc.IS_CURRENT = TRUE
    WHERE sp.BATCH_ID = :BATCH_ID
      AND (dc.CUSTOMER_KEY IS NULL  -- New customer
           OR dc.RISK_TIER != sp.RISK_TIER
           OR dc.AVG_TRANSACTION_AMOUNT != sp.AVG_TRANSACTION_AMOUNT);

    LET rows_updated := SQLROWCOUNT;

    RETURN 'Updated ' || :rows_updated || ' customer dimension records for batch ' || :BATCH_ID;
END;
$$;
