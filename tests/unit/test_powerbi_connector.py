"""Unit tests for Power BI Snowflake connector artifacts."""

from __future__ import annotations

import pytest

from dashboards.powerbi.data_connections.snowflake_connector import (
    PowerBIConnectionError,
    PowerBIRefreshFrequency,
    PowerBISnowflakeConnector,
    RefreshSchedule,
    RowLevelSecurityConfig,
    SnowflakePowerBIConfig,
    build_refresh_schedule,
    build_rls_metadata,
    build_snowflake_rls_setup_sql,
)


def _config(**overrides: object) -> SnowflakePowerBIConfig:
    values = {
        "account": "acct",
        "user": "svc_powerbi",
        "password": "secret",
        "warehouse": "WH",
        "database": "RISKPULSE",
        "lookback_days": 365,
    }
    values.update(overrides)
    return SnowflakePowerBIConfig(**values)


def test_config_validates_required_credentials() -> None:
    with pytest.raises(PowerBIConnectionError):
        SnowflakePowerBIConfig(account="", user="svc", password="secret").validate()

    with pytest.raises(PowerBIConnectionError):
        SnowflakePowerBIConfig(account="acct", user="svc").validate()


def test_connection_parameters_support_password_and_private_key() -> None:
    password_params = _config().connection_parameters()
    key_params = _config(password=None, private_key_path="C:/secure/key.pem").connection_parameters()

    assert password_params["password"] == "secret"
    assert key_params["private_key_file"] == "C:/secure/key.pem"
    assert key_params["role"] == "RISKPULSE_POWERBI_ROLE"


def test_render_query_applies_lookback_and_tenant_filter() -> None:
    connector = PowerBISnowflakeConnector(config=_config(tenant_id="tenant-a"))
    query = connector.render_query("DailyFraudSummary")

    assert "DATEADD(DAY, -365, CURRENT_DATE())" in query
    assert "WHERE TENANT_ID = 'tenant-a'" in query
    assert "DAILY_FRAUD_SUMMARY" in query


def test_power_query_generation_includes_all_expected_datasets() -> None:
    connector = PowerBISnowflakeConnector(config=_config())
    power_query = connector.generate_power_query_m()

    assert "section RiskPulsePowerBI" in power_query
    assert "ExecutiveSummary =" in power_query
    assert "FraudByGeography =" in power_query
    assert "AlertResolution =" in power_query
    assert "EnableFolding=true" in power_query


def test_refresh_schedule_payload_uses_powerbi_shape() -> None:
    schedule = build_refresh_schedule(PowerBIRefreshFrequency.DAILY)

    assert schedule["value"]["enabled"] is True
    assert schedule["value"]["notifyOption"] == "MailOnFailure"
    assert schedule["value"]["times"] == ["06:00", "12:00", "18:00"]


def test_refresh_schedule_rejects_bad_times() -> None:
    with pytest.raises(ValueError):
        RefreshSchedule(refresh_times=("25:00",)).validate()


def test_rls_metadata_builds_userprincipalname_filters() -> None:
    metadata = build_rls_metadata(["DailyFraudSummary"], RowLevelSecurityConfig())

    role = metadata["roles"][0]
    assert role["name"] == "Tenant Restricted"
    assert "USERPRINCIPALNAME()" in role["filters"]["DailyFraudSummary"]
    assert metadata["tenantAccessTable"] == "SECURITY.POWERBI_USER_TENANT_ACCESS"


def test_snowflake_rls_setup_sql_creates_security_table() -> None:
    sql = build_snowflake_rls_setup_sql()

    assert "CREATE TABLE IF NOT EXISTS SECURITY.POWERBI_USER_TENANT_ACCESS" in sql
    assert "USER_PRINCIPAL_NAME" in sql
    assert "TENANT_ID" in sql
