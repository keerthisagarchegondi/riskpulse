"""Unit tests for Streamlit dashboard component helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from dashboards.streamlit.auth.roles import (
    DashboardRole,
    can_access_page,
    parse_role,
    visible_pages_for_role,
)
from dashboards.streamlit.pages.alert_management import (
    calculate_alert_kpis,
    calculate_resolution_by_analyst,
    calculate_rule_effectiveness,
    calculate_sla_metrics,
)
from dashboards.streamlit.pages.demo_fallback import (
    demo_alerts,
    demo_model_scores,
    demo_transactions,
)
from dashboards.streamlit.pages.model_performance import (
    build_auc_trend,
    build_confusion_matrix,
    build_degradation_alerts,
    build_precision_recall_summary,
    calculate_population_stability_index,
)


def _model_scores() -> pd.DataFrame:
    base = datetime(2026, 7, 1, tzinfo=timezone.utc)
    return pd.DataFrame(
        {
            "transaction_id": [f"txn-{idx}" for idx in range(12)],
            "model_version": ["v1"] * 6 + ["v2"] * 6,
            "overall_score": [
                0.95,
                0.88,
                0.72,
                0.35,
                0.21,
                0.08,
                0.91,
                0.82,
                0.64,
                0.42,
                0.18,
                0.05,
            ],
            "actual_label": [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
            "latency_ms": [40, 50, 60, 45, 55, 65, 42, 58, 75, 63, 52, 49],
            "scoring_timestamp": [base + timedelta(hours=idx) for idx in range(12)],
        }
    )


def _alerts() -> pd.DataFrame:
    now = datetime.now(tz=timezone.utc)
    return pd.DataFrame(
        {
            "alert_id": ["a1", "a2", "a3", "a4"],
            "severity": ["critical", "high", "medium", "low"],
            "status": ["resolved", "false_positive", "open", "investigating"],
            "assigned_to": ["ana", "ana", "sam", None],
            "created_at": [
                now - timedelta(hours=1),
                now - timedelta(hours=8),
                now - timedelta(hours=2),
                now - timedelta(hours=60),
            ],
            "updated_at": [
                now - timedelta(minutes=45),
                now - timedelta(hours=7),
                now - timedelta(hours=1),
                now - timedelta(hours=50),
            ],
            "resolved_at": [
                now - timedelta(minutes=30),
                now - timedelta(hours=3),
                pd.NaT,
                pd.NaT,
            ],
            "response_hours": [0.25, 1.0, 1.0, 10.0],
            "resolution_hours": [0.5, 5.0, None, None],
        }
    )


def test_role_based_page_visibility_restricts_admin_pages() -> None:
    pages = {
        "Real Time": "real_time_monitor",
        "Model Performance": "model_performance",
        "Alert Management": "alert_management",
    }

    assert parse_role("not-a-role") == DashboardRole.VIEWER
    assert can_access_page(DashboardRole.ADMIN, "model_performance")
    assert not can_access_page(DashboardRole.ANALYST, "model_performance")
    assert visible_pages_for_role(DashboardRole.ANALYST, pages) == {
        "Real Time": "real_time_monitor"
    }


def test_confusion_matrix_uses_thresholded_scores() -> None:
    matrix = build_confusion_matrix(_model_scores(), threshold=0.7)

    assert matrix.loc["Actual Fraud", "Predicted Fraud"] == 4
    assert matrix.loc["Actual Fraud", "Predicted Legitimate"] == 2
    assert matrix.loc["Actual Legitimate", "Predicted Fraud"] == 1
    assert matrix.loc["Actual Legitimate", "Predicted Legitimate"] == 5


def test_auc_and_precision_recall_summaries_are_bounded() -> None:
    scores = _model_scores()
    auc_trend = build_auc_trend(scores, grain="day", min_labels=4)
    pr_summary = build_precision_recall_summary(scores)

    assert not auc_trend.empty
    assert auc_trend["auc"].between(0, 1).all()
    assert set(pr_summary["model_version"]) == {"v1", "v2"}
    assert pr_summary["average_precision"].between(0, 1).all()


def test_population_stability_index_detects_distribution_shift() -> None:
    baseline = pd.Series([0.05, 0.10, 0.15, 0.20, 0.25] * 20)
    current = pd.Series([0.75, 0.80, 0.85, 0.90, 0.95] * 20)

    assert calculate_population_stability_index(baseline, current) > 0.25


def test_degradation_alerts_include_latency_and_drift() -> None:
    scores = _model_scores()
    scores["latency_ms"] = [300] * len(scores.index)
    alerts = build_degradation_alerts(
        scores,
        baseline_scores=pd.Series([0.1, 0.2, 0.3] * 50),
        current_scores=pd.Series([0.8, 0.9, 0.95] * 50),
        latency_p95_ms=250,
    )

    assert {"p95_latency_ms", "score_psi"}.issubset(set(alerts["metric"]))


def test_sla_metrics_classify_met_and_breached_alerts() -> None:
    sla_df = calculate_sla_metrics(_alerts())

    assert sla_df.loc[sla_df["alert_id"] == "a1", "sla_status"].item() == "met"
    assert sla_df.loc[sla_df["alert_id"] == "a4", "sla_status"].item() == "breached_open"


def test_alert_kpis_and_analyst_resolution_metrics() -> None:
    alerts = _alerts()
    kpis = calculate_alert_kpis(alerts)
    analyst_df = calculate_resolution_by_analyst(alerts)

    assert kpis["total_alerts"] == 4
    assert kpis["resolution_rate"] == pytest.approx(0.5)
    assert kpis["false_positive_rate"] == pytest.approx(0.5)
    ana = analyst_df[analyst_df["assigned_to"] == "ana"].iloc[0]
    assert ana["closed_alerts"] == 2
    assert ana["resolution_rate"] == pytest.approx(1.0)


def test_rule_effectiveness_flags_noisy_rules() -> None:
    rules = pd.DataFrame(
        {
            "rule_id": ["R1", "R2"],
            "rule_name": ["Velocity", "Geo"],
            "rule_category": ["velocity", "geo"],
            "triggered_count": [20, 20],
            "confirmed_count": [3, 18],
            "false_positive_count": [12, 2],
            "closed_count": [15, 20],
            "avg_resolution_hours": [4.0, 2.0],
        }
    )

    result = calculate_rule_effectiveness(rules)

    assert result.loc[result["rule_id"] == "R1", "action_hint"].item() == "review_threshold"
    assert result.loc[result["rule_id"] == "R2", "action_hint"].item() == "high_value"


def test_demo_fallback_data_matches_dashboard_contracts() -> None:
    txns = demo_transactions()
    scores = demo_model_scores()
    alerts = demo_alerts()

    assert {
        "transaction_id",
        "transaction_amount",
        "status",
        "risk_score",
        "channel",
        "geo_country",
        "transaction_timestamp",
    }.issubset(txns.columns)
    assert {"overall_score", "actual_label", "model_version", "latency_ms"}.issubset(scores.columns)
    assert {"alert_id", "severity", "status", "assigned_to", "created_at"}.issubset(alerts.columns)
    assert txns["risk_score"].between(0, 1).all()
    assert scores["overall_score"].between(0, 1).all()
    assert 0 < (txns["status"] == "flagged").mean() < 0.15
    assert calculate_alert_kpis(alerts)["total_alerts"] > 0
