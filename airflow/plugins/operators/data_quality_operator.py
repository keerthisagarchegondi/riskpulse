"""Custom data quality operator for Airflow.

Runs configurable data quality checks (completeness, freshness, volume
anomaly detection) against the RiskPulse operational database or data
warehouse and publishes a quality report.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Sequence

import sqlalchemy as sa
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.models import BaseOperator
from airflow.utils.context import Context

from src.utils.logger import get_logger

logger = get_logger(__name__, component="data_quality_operator")


class CheckSeverity(str, Enum):
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CheckResult:
    """Result of a single data quality check."""

    check_name: str
    passed: bool
    severity: str
    metric_value: Any = None
    threshold: Any = None
    details: str = ""


@dataclass
class QualityReport:
    """Aggregated quality report for a single run."""

    run_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_checks: int = 0
    passed: int = 0
    warnings: int = 0
    critical_failures: int = 0
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.critical_failures == 0

    def add(self, result: CheckResult) -> None:
        self.total_checks += 1
        if result.passed:
            self.passed += 1
        elif result.severity == CheckSeverity.CRITICAL:
            self.critical_failures += 1
        else:
            self.warnings += 1
        self.checks.append(
            {
                "check_name": result.check_name,
                "passed": result.passed,
                "severity": result.severity,
                "metric_value": result.metric_value,
                "threshold": result.threshold,
                "details": result.details,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_timestamp": self.run_timestamp,
            "total_checks": self.total_checks,
            "passed": self.passed,
            "warnings": self.warnings,
            "critical_failures": self.critical_failures,
            "all_passed": self.all_passed,
            "checks": self.checks,
        }


# Default check configurations
_DEFAULT_COMPLETENESS_CHECKS: list[dict[str, Any]] = [
    {
        "table": "transactions",
        "column": "transaction_id",
        "severity": CheckSeverity.CRITICAL,
    },
    {
        "table": "transactions",
        "column": "amount",
        "severity": CheckSeverity.CRITICAL,
    },
    {
        "table": "transactions",
        "column": "account_id",
        "severity": CheckSeverity.CRITICAL,
    },
    {
        "table": "transactions",
        "column": "merchant_name",
        "severity": CheckSeverity.WARNING,
    },
]

_DEFAULT_FRESHNESS_CHECKS: list[dict[str, Any]] = [
    {
        "table": "transactions",
        "timestamp_column": "created_at",
        "max_delay_minutes": 30,
        "severity": CheckSeverity.CRITICAL,
    },
    {
        "table": "fraud_alerts",
        "timestamp_column": "created_at",
        "max_delay_minutes": 60,
        "severity": CheckSeverity.WARNING,
    },
]

_VOLUME_LOOKBACK_DAYS = 7
_VOLUME_STDDEV_THRESHOLD = 2.0


class DataQualityOperator(BaseOperator):
    """Run data quality checks against the operational database.

    Supports three categories of checks:

    1. **Completeness** – verifies required columns have no/low NULL rates.
    2. **Freshness** – verifies latest data is within an acceptable delay.
    3. **Volume anomaly** – detects unusual row count deviations compared to
       the trailing window.

    The operator pushes a ``QualityReport`` dict to XCom and raises
    ``AirflowException`` when any *critical* check fails.

    Parameters
    ----------
    conn_id : str
        Airflow connection ID for the target database.
    completeness_checks : list[dict] | None
        Override default completeness check definitions.
    freshness_checks : list[dict] | None
        Override default freshness check definitions.
    volume_tables : list[str] | None
        Tables to run volume anomaly detection on.
    volume_lookback_days : int
        Number of trailing days for the volume baseline.
    volume_stddev_threshold : float
        Z-score threshold above which a volume is anomalous.
    max_null_rate : float
        Maximum allowed NULL fraction for completeness checks (0.0–1.0).
    fail_on_warning : bool
        If ``True``, also fail the task on WARNING-level check failures.
    """

    template_fields: Sequence[str] = ("conn_id",)

    def __init__(
        self,
        *,
        conn_id: str = "riskpulse_postgres",
        completeness_checks: list[dict[str, Any]] | None = None,
        freshness_checks: list[dict[str, Any]] | None = None,
        volume_tables: list[str] | None = None,
        volume_lookback_days: int = _VOLUME_LOOKBACK_DAYS,
        volume_stddev_threshold: float = _VOLUME_STDDEV_THRESHOLD,
        max_null_rate: float = 0.01,
        fail_on_warning: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.conn_id = conn_id
        self.completeness_checks = completeness_checks or _DEFAULT_COMPLETENESS_CHECKS
        self.freshness_checks = freshness_checks or _DEFAULT_FRESHNESS_CHECKS
        self.volume_tables = volume_tables or ["transactions", "fraud_alerts"]
        self.volume_lookback_days = volume_lookback_days
        self.volume_stddev_threshold = volume_stddev_threshold
        self.max_null_rate = max_null_rate
        self.fail_on_warning = fail_on_warning

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, context: Context) -> dict[str, Any]:
        report = QualityReport()
        engine = self._get_engine()

        with engine.connect() as conn:
            self._run_completeness_checks(conn, report)
            self._run_freshness_checks(conn, report)
            self._run_volume_checks(conn, report)

        summary = report.to_dict()
        logger.info(
            "Data quality checks completed",
            total=report.total_checks,
            passed=report.passed,
            warnings=report.warnings,
            critical=report.critical_failures,
        )

        if report.critical_failures > 0:
            raise AirflowException(
                f"Data quality: {report.critical_failures} critical "
                f"check(s) failed. Report: {summary}"
            )

        if self.fail_on_warning and report.warnings > 0:
            raise AirflowException(
                f"Data quality: {report.warnings} warning(s) "
                f"with fail_on_warning=True. Report: {summary}"
            )

        return summary

    # ------------------------------------------------------------------
    # Check implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_identifier(name: str) -> str:
        """Validate that a SQL identifier contains only safe characters."""
        import re

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(f"Invalid SQL identifier: {name!r}")
        return name

    def _run_completeness_checks(self, conn: sa.engine.Connection, report: QualityReport) -> None:
        for check in self.completeness_checks:
            table = self._validate_identifier(check["table"])
            column = self._validate_identifier(check["column"])
            severity = check.get("severity", CheckSeverity.WARNING)

            sql = (
                f"SELECT COUNT(*) AS total, "  # nosec B608
                f"SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) AS nulls "
                f"FROM {table}"
            )  # nosec B608
            result = conn.execute(sa.text(sql))
            row = result.mappings().fetchone()
            total = row["total"] if row else 0
            nulls = row["nulls"] if row else 0
            null_rate = nulls / total if total > 0 else 0.0

            passed = null_rate <= self.max_null_rate
            report.add(
                CheckResult(
                    check_name=f"completeness_{table}_{column}",
                    passed=passed,
                    severity=severity,
                    metric_value=round(null_rate, 6),
                    threshold=self.max_null_rate,
                    details=(f"{nulls}/{total} nulls ({null_rate:.4%})" if not passed else "OK"),
                )
            )
            logger.debug(
                "Completeness check",
                table=table,
                column=column,
                null_rate=round(null_rate, 6),
                passed=passed,
            )

    def _run_freshness_checks(self, conn: sa.engine.Connection, report: QualityReport) -> None:
        now = datetime.now(timezone.utc)
        for check in self.freshness_checks:
            table = self._validate_identifier(check["table"])
            ts_col = self._validate_identifier(check["timestamp_column"])
            max_delay = check["max_delay_minutes"]
            severity = check.get("severity", CheckSeverity.WARNING)

            sql = f"SELECT MAX({ts_col}) AS latest FROM {table}"  # nosec B608
            result = conn.execute(sa.text(sql))
            row = result.mappings().fetchone()
            latest = row["latest"] if row else None

            if latest is None:
                report.add(
                    CheckResult(
                        check_name=f"freshness_{table}",
                        passed=False,
                        severity=severity,
                        metric_value=None,
                        threshold=max_delay,
                        details=f"No rows in {table}",
                    )
                )
                continue

            if latest.tzinfo is None:
                latest = latest.replace(tzinfo=timezone.utc)

            delay_minutes = (now - latest).total_seconds() / 60.0
            passed = delay_minutes <= max_delay

            report.add(
                CheckResult(
                    check_name=f"freshness_{table}",
                    passed=passed,
                    severity=severity,
                    metric_value=round(delay_minutes, 2),
                    threshold=max_delay,
                    details=(
                        f"Latest record {delay_minutes:.1f}min ago (limit {max_delay}min)"
                        if not passed
                        else "OK"
                    ),
                )
            )
            logger.debug(
                "Freshness check",
                table=table,
                delay_minutes=round(delay_minutes, 2),
                passed=passed,
            )

    def _run_volume_checks(self, conn: sa.engine.Connection, report: QualityReport) -> None:
        now = datetime.now(timezone.utc)

        for raw_table in self.volume_tables:
            table = self._validate_identifier(raw_table)
            # Fetch daily counts for the lookback window + today
            sql = (
                f"SELECT DATE(created_at) AS day, COUNT(*) AS cnt "  # nosec B608
                f"FROM {table} "
                f"WHERE created_at >= :start "
                f"GROUP BY DATE(created_at) "
                f"ORDER BY day"
            )  # nosec B608
            result = conn.execute(
                sa.text(sql),
                {"start": now - timedelta(days=self.volume_lookback_days + 1)},
            )
            rows = result.mappings().fetchall()

            if len(rows) < 2:
                report.add(
                    CheckResult(
                        check_name=f"volume_{table}",
                        passed=True,
                        severity=CheckSeverity.WARNING,
                        details="Insufficient history for anomaly detection",
                    )
                )
                continue

            historical = [r["cnt"] for r in rows[:-1]]
            today_count = rows[-1]["cnt"]
            mean = statistics.mean(historical)
            stdev = statistics.stdev(historical) if len(historical) > 1 else 0.0

            if stdev == 0:
                z_score = 0.0
            else:
                z_score = abs(today_count - mean) / stdev

            passed = z_score <= self.volume_stddev_threshold
            report.add(
                CheckResult(
                    check_name=f"volume_{table}",
                    passed=passed,
                    severity=CheckSeverity.WARNING,
                    metric_value={
                        "today": today_count,
                        "mean": round(mean, 1),
                        "z_score": round(z_score, 2),
                    },
                    threshold=self.volume_stddev_threshold,
                    details=(
                        f"z={z_score:.2f} (today={today_count}, mean={mean:.1f})"
                        if not passed
                        else "OK"
                    ),
                )
            )
            logger.debug(
                "Volume check",
                table=table,
                today=today_count,
                mean=round(mean, 1),
                z_score=round(z_score, 2),
                passed=passed,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_engine(self) -> sa.engine.Engine:
        hook = BaseHook.get_hook(self.conn_id)
        return hook.get_sqlalchemy_engine()
