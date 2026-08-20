"""Snowflake extraction and Power BI configuration helpers.

This module supports the executive Power BI dashboards with:
- bounded Snowflake extraction queries for Import mode
- Power Query M generation for DirectQuery or Import datasets
- scheduled refresh configuration payloads
- row-level security metadata for tenant-restricted reports
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import time
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a project dependency
    pd = None  # type: ignore[assignment]

try:
    import snowflake.connector
    from snowflake.connector import DictCursor
except ImportError:  # pragma: no cover - connector is optional for artifact generation
    snowflake = None  # type: ignore[assignment]

    class DictCursor:  # type: ignore[no-redef]
        pass


class PowerBIConnectionError(RuntimeError):
    """Raised when Power BI Snowflake connection configuration is invalid."""


_SNOWFLAKE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _validate_snowflake_identifier(identifier: str, *, name: str = "identifier") -> str:
    """Validate Snowflake identifiers before embedding them in generated SQL."""
    if not isinstance(identifier, str) or not _SNOWFLAKE_IDENTIFIER_RE.fullmatch(identifier):
        raise PowerBIConnectionError(f"Invalid Snowflake {name}: {identifier!r}")
    return identifier


def _validate_qualified_table_name(value: str) -> tuple[str, str]:
    parts = value.split(".")
    if len(parts) != 2:
        raise PowerBIConnectionError("RLS access table must be schema-qualified")
    schema = _validate_snowflake_identifier(parts[0], name="schema")
    table = _validate_snowflake_identifier(parts[1], name="table")
    return schema, table


def _snowflake_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class PowerBIRefreshFrequency(StrEnum):
    """Supported refresh schedule cadences."""

    DAILY = "daily"
    HOURLY = "hourly"


@dataclass(frozen=True)
class RefreshSchedule:
    """Power BI refresh schedule settings."""

    frequency: PowerBIRefreshFrequency = PowerBIRefreshFrequency.DAILY
    enabled: bool = True
    local_time_zone_id: str = "Eastern Standard Time"
    refresh_times: tuple[str, ...] = ("06:00", "12:00", "18:00")
    notify_option: str = "MailOnFailure"

    def validate(self) -> None:
        """Validate refresh times use HH:MM format."""
        for value in self.refresh_times:
            try:
                time.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"Invalid refresh time: {value}") from exc

    def to_powerbi_payload(self) -> dict[str, Any]:
        """Return a Power BI REST API compatible refresh schedule payload."""
        self.validate()
        return {
            "value": {
                "enabled": self.enabled,
                "localTimeZoneId": self.local_time_zone_id,
                "notifyOption": self.notify_option,
                "times": list(self.refresh_times),
            }
        }


@dataclass(frozen=True)
class RowLevelSecurityConfig:
    """Power BI row-level security configuration."""

    tenant_column: str = "TENANT_ID"
    user_column: str = "USER_PRINCIPAL_NAME"
    access_table: str = "SECURITY.POWERBI_USER_TENANT_ACCESS"
    default_role_name: str = "Tenant Restricted"
    admin_role_name: str = "Executive Admin"
    admin_group: str = "RiskPulse Executive Admins"

    def dax_filter(self, table_name: str) -> str:
        """Return the DAX table filter expression for a tenant-scoped table."""
        return (
            f"{table_name}[{self.tenant_column}] IN "
            f"CALCULATETABLE(VALUES(TenantAccess[{self.tenant_column}]), "
            f"TenantAccess[{self.user_column}] = USERPRINCIPALNAME())"
        )


@dataclass(frozen=True)
class SnowflakePowerBIConfig:
    """Snowflake connection settings for Power BI dashboard datasets."""

    account: str
    user: str
    password: str | None = None
    private_key_path: str | None = None
    warehouse: str = "RISKPULSE_WH"
    database: str = "RISKPULSE"
    role: str = "RISKPULSE_POWERBI_ROLE"
    schema: str = "REPORTING"
    lookback_days: int = 730
    tenant_id: str | None = None
    authenticator: str | None = None

    @classmethod
    def from_env(cls) -> "SnowflakePowerBIConfig":
        """Build configuration from environment variables."""
        return cls(
            account=os.environ.get("POWERBI_SNOWFLAKE_ACCOUNT")
            or os.environ.get("SNOWFLAKE_ACCOUNT", ""),
            user=os.environ.get("POWERBI_SNOWFLAKE_USER") or os.environ.get("SNOWFLAKE_USER", ""),
            password=os.environ.get("POWERBI_SNOWFLAKE_PASSWORD")
            or os.environ.get("SNOWFLAKE_PASSWORD"),
            private_key_path=os.environ.get("POWERBI_SNOWFLAKE_PRIVATE_KEY_PATH")
            or os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH"),
            warehouse=os.environ.get("POWERBI_SNOWFLAKE_WAREHOUSE")
            or os.environ.get("SNOWFLAKE_WAREHOUSE", "RISKPULSE_WH"),
            database=os.environ.get("POWERBI_SNOWFLAKE_DATABASE")
            or os.environ.get("SNOWFLAKE_DATABASE", "RISKPULSE"),
            role=os.environ.get("POWERBI_SNOWFLAKE_ROLE")
            or os.environ.get("SNOWFLAKE_ROLE", "RISKPULSE_POWERBI_ROLE"),
            schema=os.environ.get("POWERBI_SNOWFLAKE_SCHEMA", "REPORTING"),
            lookback_days=int(os.environ.get("POWERBI_LOOKBACK_DAYS", "730")),
            tenant_id=os.environ.get("POWERBI_TENANT_ID"),
            authenticator=os.environ.get("POWERBI_SNOWFLAKE_AUTHENTICATOR"),
        )

    def validate(self) -> None:
        """Validate required connection settings."""
        if not self.account:
            raise PowerBIConnectionError("Snowflake account is required.")
        if not self.user:
            raise PowerBIConnectionError("Snowflake user is required.")
        if not self.password and not self.private_key_path and not self.authenticator:
            raise PowerBIConnectionError(
                "Snowflake password, private key path, or authenticator is required."
            )

    def connection_parameters(self) -> dict[str, Any]:
        """Return parameters accepted by snowflake.connector.connect."""
        self.validate()
        params: dict[str, Any] = {
            "account": self.account,
            "user": self.user,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema,
            "role": self.role,
        }
        if self.authenticator:
            params["authenticator"] = self.authenticator
        elif self.private_key_path:
            params["private_key_file"] = self.private_key_path
        else:
            params["password"] = self.password
        return params


QUERY_MAP: dict[str, str] = {
    "ExecutiveSummary": """
        SELECT
            WEEK_START_DATE,
            WEEK_END_DATE,
            TOTAL_TRANSACTIONS,
            TOTAL_AMOUNT,
            FRAUD_PREVENTED_COUNT,
            FRAUD_PREVENTED_AMOUNT,
            FRAUD_RATE_CHANGE_PCT,
            MODEL_ACCURACY,
            SYSTEM_UPTIME_PCT,
            AVG_DETECTION_LATENCY_MS,
            ALERTS_TOTAL,
            ALERTS_RESOLVED,
            FALSE_POSITIVE_RATE,
            TOP_FRAUD_CATEGORY,
            TOP_FRAUD_COUNTRY,
            'default' AS TENANT_ID
        FROM REPORTING.WEEKLY_EXECUTIVE_SUMMARY
        WHERE WEEK_START_DATE >= DATEADD(DAY, -%(lookback_days)s, CURRENT_DATE())
    """,
    "DailyFraudSummary": """
        SELECT
            SUMMARY_DATE,
            TOTAL_TRANSACTIONS,
            TOTAL_AMOUNT,
            FRAUD_TRANSACTIONS,
            FRAUD_AMOUNT,
            FRAUD_RATE,
            AVG_RISK_SCORE,
            MEDIAN_RISK_SCORE,
            HIGH_RISK_COUNT,
            CRITICAL_RISK_COUNT,
            ALERTS_GENERATED,
            ALERTS_RESOLVED,
            FALSE_POSITIVE_COUNT,
            FALSE_POSITIVE_RATE,
            AVG_RESOLUTION_TIME_MINUTES,
            UNIQUE_ACCOUNTS_FLAGGED,
            'default' AS TENANT_ID
        FROM REPORTING.DAILY_FRAUD_SUMMARY
        WHERE SUMMARY_DATE >= DATEADD(DAY, -%(lookback_days)s, CURRENT_DATE())
    """,
    "FraudByGeography": """
        SELECT
            SUMMARY_DATE,
            COUNTRY_CODE,
            COUNTRY_NAME,
            TOTAL_TRANSACTIONS,
            FRAUD_TRANSACTIONS,
            FRAUD_RATE,
            TOTAL_AMOUNT,
            FRAUD_AMOUNT,
            AVG_RISK_SCORE,
            'default' AS TENANT_ID
        FROM REPORTING.FRAUD_BY_GEOGRAPHY
        WHERE SUMMARY_DATE >= DATEADD(DAY, -%(lookback_days)s, CURRENT_DATE())
    """,
    "FraudByMerchantCategory": """
        SELECT
            SUMMARY_DATE,
            CATEGORY_CODE,
            CATEGORY_NAME,
            TOTAL_TRANSACTIONS,
            FRAUD_TRANSACTIONS,
            FRAUD_RATE,
            TOTAL_AMOUNT,
            FRAUD_AMOUNT,
            AVG_RISK_SCORE,
            'default' AS TENANT_ID
        FROM REPORTING.FRAUD_BY_MERCHANT_CATEGORY
        WHERE SUMMARY_DATE >= DATEADD(DAY, -%(lookback_days)s, CURRENT_DATE())
    """,
    "FraudByChannel": """
        SELECT
            METRIC_HOUR::DATE AS SUMMARY_DATE,
            'online' AS CHANNEL,
            SUM(CHANNEL_ONLINE_COUNT) AS TOTAL_TRANSACTIONS,
            SUM(FRAUD_COUNT) AS FRAUD_TRANSACTIONS,
            AVG(AVG_RISK_SCORE) AS AVG_RISK_SCORE,
            'default' AS TENANT_ID
        FROM REPORTING.HOURLY_TRANSACTION_METRICS
        WHERE METRIC_HOUR::DATE >= DATEADD(DAY, -%(lookback_days)s, CURRENT_DATE())
        GROUP BY METRIC_HOUR::DATE
        UNION ALL
        SELECT METRIC_HOUR::DATE, 'pos', SUM(CHANNEL_POS_COUNT), SUM(FRAUD_COUNT),
            AVG(AVG_RISK_SCORE), 'default'
        FROM REPORTING.HOURLY_TRANSACTION_METRICS
        WHERE METRIC_HOUR::DATE >= DATEADD(DAY, -%(lookback_days)s, CURRENT_DATE())
        GROUP BY METRIC_HOUR::DATE
        UNION ALL
        SELECT METRIC_HOUR::DATE, 'atm', SUM(CHANNEL_ATM_COUNT), SUM(FRAUD_COUNT),
            AVG(AVG_RISK_SCORE), 'default'
        FROM REPORTING.HOURLY_TRANSACTION_METRICS
        WHERE METRIC_HOUR::DATE >= DATEADD(DAY, -%(lookback_days)s, CURRENT_DATE())
        GROUP BY METRIC_HOUR::DATE
        UNION ALL
        SELECT METRIC_HOUR::DATE, 'mobile', SUM(CHANNEL_MOBILE_COUNT), SUM(FRAUD_COUNT),
            AVG(AVG_RISK_SCORE), 'default'
        FROM REPORTING.HOURLY_TRANSACTION_METRICS
        WHERE METRIC_HOUR::DATE >= DATEADD(DAY, -%(lookback_days)s, CURRENT_DATE())
        GROUP BY METRIC_HOUR::DATE
    """,
    "OperationalMetrics": """
        SELECT
            METRIC_HOUR,
            TOTAL_TRANSACTIONS,
            TOTAL_AMOUNT,
            AVG_AMOUNT,
            FRAUD_COUNT,
            HIGH_RISK_COUNT,
            AVG_RISK_SCORE,
            AVG_LATENCY_MS,
            ERROR_COUNT,
            'default' AS TENANT_ID
        FROM REPORTING.HOURLY_TRANSACTION_METRICS
        WHERE METRIC_HOUR::DATE >= DATEADD(DAY, -%(lookback_days)s, CURRENT_DATE())
    """,
    "ModelPerformance": """
        SELECT
            SUMMARY_DATE,
            MODEL_NAME,
            MODEL_VERSION,
            PREDICTIONS_COUNT,
            AVG_SCORE,
            PRECISION_SCORE,
            RECALL_SCORE,
            F1_SCORE,
            AUC_ROC,
            AVG_LATENCY_MS,
            P95_LATENCY_MS,
            'default' AS TENANT_ID
        FROM REPORTING.MODEL_PERFORMANCE_DAILY
        WHERE SUMMARY_DATE >= DATEADD(DAY, -%(lookback_days)s, CURRENT_DATE())
    """,
    "AlertResolution": """
        SELECT
            SUMMARY_DATE,
            SEVERITY,
            ALERTS_CREATED,
            ALERTS_RESOLVED,
            ALERTS_ESCALATED,
            FALSE_POSITIVES,
            AVG_RESOLUTION_TIME_MINUTES,
            MEDIAN_RESOLUTION_TIME_MINUTES,
            P95_RESOLUTION_TIME_MINUTES,
            SLA_COMPLIANCE_RATE,
            'default' AS TENANT_ID
        FROM REPORTING.ALERT_RESOLUTION_METRICS
        WHERE SUMMARY_DATE >= DATEADD(DAY, -%(lookback_days)s, CURRENT_DATE())
    """,
    "TenantAccess": """
        SELECT
            USER_PRINCIPAL_NAME,
            TENANT_ID,
            ACCESS_ROLE
        FROM SECURITY.POWERBI_USER_TENANT_ACCESS
        WHERE IS_ACTIVE = TRUE
    """,
}


@dataclass
class PowerBISnowflakeConnector:
    """Snowflake connector for Power BI executive analytics datasets."""

    config: SnowflakePowerBIConfig = field(default_factory=SnowflakePowerBIConfig.from_env)
    query_map: dict[str, str] = field(default_factory=lambda: QUERY_MAP.copy())
    _connection: Any | None = field(default=None, init=False, repr=False)

    def connect(self) -> None:
        """Open a Snowflake connection."""
        if snowflake is None:
            raise PowerBIConnectionError("snowflake-connector-python is not installed.")
        self._connection = snowflake.connector.connect(
            **self.config.connection_parameters(),
            client_session_keep_alive=True,
        )

    def close(self) -> None:
        """Close the Snowflake connection."""
        if self._connection is not None:
            self._connection.close()
        self._connection = None

    def __enter__(self) -> "PowerBISnowflakeConnector":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def render_query(self, dataset_name: str) -> str:
        """Render a configured dataset query with lookback and tenant predicates."""
        if dataset_name not in self.query_map:
            raise KeyError(f"Unknown Power BI dataset: {dataset_name}")

        query = self.query_map[dataset_name] % {"lookback_days": self.config.lookback_days}
        if self.config.tenant_id and "TENANT_ID" in query.upper():
            tenant_id = _snowflake_string_literal(self.config.tenant_id)
            query = f"SELECT * FROM ({query}) WHERE TENANT_ID = {tenant_id}"  # nosec B608
        return query.strip()

    def fetch_dataframe(self, dataset_name: str) -> Any:
        """Fetch a single dataset as a pandas DataFrame."""
        if pd is None:
            raise PowerBIConnectionError("pandas is required for dataframe extraction.")
        if self._connection is None:
            self.connect()

        query = self.render_query(dataset_name)
        cursor = self._connection.cursor(DictCursor)
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
        finally:
            cursor.close()

        return pd.DataFrame(rows)

    def extract_all(self, datasets: Iterable[str] | None = None) -> dict[str, Any]:
        """Extract all configured dashboard datasets."""
        names = list(datasets or self.query_map.keys())
        return {name: self.fetch_dataframe(name) for name in names}

    def validate_sources(self, required: Iterable[str] | None = None) -> dict[str, bool]:
        """Validate that required source queries return successfully."""
        required_names = list(required or self.query_map.keys())
        results: dict[str, bool] = {}
        for name in required_names:
            try:
                df = self.fetch_dataframe(name)
                results[name] = df is not None
            except Exception:
                results[name] = False
        return results

    def write_power_query_file(self, output_path: Path) -> None:
        """Write Power Query M definitions for all configured datasets."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.generate_power_query_m(), encoding="utf-8")

    def generate_power_query_m(self) -> str:
        """Generate Power Query M definitions using Snowflake native query folding."""
        sections: list[str] = []
        for name in self.query_map:
            escaped_query = self.render_query(name).replace('"', '""').replace("\n", " ")
            sections.append(
                f"{name} = Value.NativeQuery("
                f'Snowflake.Databases("{self.config.account}", "{self.config.warehouse}")'
                f'{{[Name="{self.config.database}"]}}[Data], '
                f'"{escaped_query}", null, [EnableFolding=true])'
            )
        return "section RiskPulsePowerBI;\n\n" + ";\n\n".join(sections) + ";\n"


def build_refresh_schedule(
    frequency: PowerBIRefreshFrequency = PowerBIRefreshFrequency.DAILY,
) -> dict[str, Any]:
    """Build the configured Power BI refresh schedule payload."""
    schedule = RefreshSchedule(
        frequency=frequency,
        refresh_times=(
            ("06:00", "12:00", "18:00")
            if frequency == PowerBIRefreshFrequency.DAILY
            else tuple(f"{hour:02d}:00" for hour in range(24))
        ),
    )
    return schedule.to_powerbi_payload()


def build_rls_metadata(
    tables: Iterable[str],
    config: RowLevelSecurityConfig | None = None,
) -> dict[str, Any]:
    """Build row-level security role metadata for a Power BI semantic model."""
    rls = config or RowLevelSecurityConfig()
    return {
        "roles": [
            {
                "name": rls.default_role_name,
                "filters": {table: rls.dax_filter(table) for table in tables},
            },
            {
                "name": rls.admin_role_name,
                "members": [rls.admin_group],
                "filters": {},
            },
        ],
        "tenantAccessTable": rls.access_table,
        "userColumn": rls.user_column,
        "tenantColumn": rls.tenant_column,
    }


def build_snowflake_rls_setup_sql(config: RowLevelSecurityConfig | None = None) -> str:
    """Return Snowflake SQL for the Power BI tenant access table."""
    rls = config or RowLevelSecurityConfig()
    schema, table = _validate_qualified_table_name(rls.access_table)
    access_table = f"{schema}.{table}"
    return f"""
CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {access_table} (
    USER_PRINCIPAL_NAME VARCHAR(255) NOT NULL,
    TENANT_ID VARCHAR(128) NOT NULL,
    ACCESS_ROLE VARCHAR(64) NOT NULL DEFAULT 'viewer',
    IS_ACTIVE BOOLEAN NOT NULL DEFAULT TRUE,
    CREATED_AT TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    UPDATED_AT TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (USER_PRINCIPAL_NAME, TENANT_ID)
);

CREATE OR REPLACE VIEW {schema}.VW_{table}_ACTIVE AS
SELECT USER_PRINCIPAL_NAME, TENANT_ID, ACCESS_ROLE
FROM {access_table}
WHERE IS_ACTIVE = TRUE;
""".strip()  # nosec B608


def write_artifacts(output_dir: Path) -> None:
    """Write generated Power BI connection artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    connector = PowerBISnowflakeConnector()
    connector.write_power_query_file(output_dir / "models" / "snowflake_queries.pq")
    (output_dir / "refresh_schedule.json").write_text(
        json.dumps(build_refresh_schedule(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "models" / "rls_roles.json").write_text(
        json.dumps(
            build_rls_metadata(
                [
                    "ExecutiveSummary",
                    "DailyFraudSummary",
                    "FraudByGeography",
                    "FraudByMerchantCategory",
                    "FraudByChannel",
                    "OperationalMetrics",
                    "ModelPerformance",
                    "AlertResolution",
                ]
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "data_connections" / "snowflake_rls_setup.sql").write_text(
        build_snowflake_rls_setup_sql(),
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_artifacts(Path(__file__).resolve().parents[1])
