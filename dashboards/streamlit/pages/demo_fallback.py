"""Local preview dashboards for database-offline Streamlit sessions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboards.streamlit.components.charts import (
    alert_severity_pie,
    fraud_rate_gauge,
    kpi_card_html,
    live_feed_table_html,
    risk_score_histogram,
    transaction_volume_chart,
)
from dashboards.streamlit.pages.alert_management import (
    calculate_alert_kpis,
    calculate_resolution_by_analyst,
    calculate_rule_effectiveness,
    calculate_sla_metrics,
)
from dashboards.streamlit.pages.model_performance import (
    build_auc_trend,
    build_confusion_matrix,
    build_degradation_alerts,
    build_precision_recall_summary,
    calculate_population_stability_index,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def demo_transactions(now: datetime | None = None) -> pd.DataFrame:
    """Build deterministic transaction preview data for local UI review."""
    base = now or datetime.now(timezone.utc).replace(second=0, microsecond=0)
    channels = ["online", "pos", "atm", "online", "pos", "online"]
    countries = ["USA", "CAN", "GBR", "MEX", "USA", "BRA"]
    scores = [0.08, 0.28, 0.22, 0.56, 0.34, 0.41]
    amounts = [42.18, 1250.0, 88.4, 415.72, 2980.5, 146.2]

    rows: list[dict[str, Any]] = []
    for idx in range(48):
        offset = idx % len(channels)
        flagged = idx in {3, 14, 27, 38}
        declined = idx in {8, 31}
        status = "flagged" if flagged else "declined" if declined else "approved"
        risk_score = 0.88 + (idx % 3) * 0.03 if flagged else scores[offset] + (idx % 5) * 0.012
        rows.append(
            {
                "transaction_id": f"demo-txn-{idx + 1:04d}",
                "transaction_amount": amounts[offset] + idx * 3.15,
                "status": status,
                "risk_score": min(0.99, risk_score),
                "overall_score": min(0.99, risk_score),
                "channel": channels[offset],
                "geo_country": countries[offset],
                "transaction_timestamp": base - timedelta(minutes=idx * 5),
            }
        )
    return pd.DataFrame(rows)


def demo_model_scores(now: datetime | None = None) -> pd.DataFrame:
    """Build deterministic labeled model score data for chart previews."""
    base = now or datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []

    for idx in range(160):
        label = 1 if idx % 7 in {0, 1} else 0
        model_version = "v2.1" if idx >= 80 else "v2.0"
        hard_case = idx % 19 == 0 or idx % 23 == 0
        center = (0.66 if model_version == "v2.0" else 0.70) if label else 0.31
        score = float(np.clip(rng.normal(center, 0.18), 0.01, 0.99))
        if hard_case:
            score = float(np.clip(1 - score + rng.normal(0, 0.04), 0.01, 0.99))
        rows.append(
            {
                "transaction_id": f"demo-score-{idx + 1:04d}",
                "model_version": model_version,
                "overall_score": score,
                "ml_score": score,
                "actual_label": label,
                "latency_ms": float(np.clip(rng.normal(62, 18), 20, 180)),
                "scoring_timestamp": base - timedelta(hours=160 - idx),
            }
        )

    return pd.DataFrame(rows)


def demo_alerts(now: datetime | None = None) -> pd.DataFrame:
    """Build deterministic alert lifecycle data for management previews."""
    base = now or datetime.now(timezone.utc).replace(second=0, microsecond=0)
    severities = ["critical", "high", "medium", "low"]
    statuses = ["resolved", "false_positive", "open", "investigating", "resolved"]
    analysts = ["ana", "sam", "lee", "maya", None]

    rows: list[dict[str, Any]] = []
    for idx in range(60):
        severity = severities[idx % len(severities)]
        status = statuses[idx % len(statuses)]
        created_at = base - timedelta(hours=idx * 6)
        updated_at = created_at + timedelta(minutes=20 + (idx % 8) * 15)
        resolved_at = None
        if status in {"resolved", "false_positive"}:
            resolved_at = created_at + timedelta(hours=1 + (idx % 10))
        rows.append(
            {
                "alert_id": f"demo-alert-{idx + 1:04d}",
                "alert_type": "rule_trigger",
                "rule_id": f"R{(idx % 6) + 1}",
                "rule_name": [
                    "High Value International",
                    "Velocity Spike",
                    "Risky Merchant",
                    "Device Mismatch",
                    "Geo Anomaly",
                    "New Account Burst",
                ][idx % 6],
                "rule_category": ["amount", "velocity", "merchant", "device", "geo", "account"][
                    idx % 6
                ],
                "severity": severity,
                "status": status,
                "assigned_to": analysts[idx % len(analysts)],
                "created_at": created_at,
                "updated_at": updated_at,
                "resolved_at": resolved_at,
                "risk_score": float(np.clip(0.35 + (idx % 9) * 0.07, 0.0, 0.98)),
                "response_hours": (updated_at - created_at).total_seconds() / 3600,
                "resolution_hours": (
                    (resolved_at - created_at).total_seconds() / 3600 if resolved_at else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def render_preview_banner(page_key: str) -> None:
    """Show a clear banner that preview data is synthetic."""
    st.markdown(
        f"""
        <div class="dashboard-feedback preview">
            <strong>Preview data mode.</strong><br>
            PostgreSQL is offline, so this {page_key.replace("_", " ")} view is using
            deterministic synthetic data for layout and chart review only.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_page(page_key: str) -> None:
    """Render page-specific local preview charts when live data is unavailable."""
    render_preview_banner(page_key)
    if page_key == "model_performance":
        _render_model_performance_preview()
    elif page_key == "alert_management":
        _render_alert_management_preview()
    elif page_key == "real_time_monitor":
        _render_real_time_preview()
    elif page_key == "trend_analysis":
        _render_trend_preview()
    elif page_key == "investigation_console":
        _render_investigation_preview()
    else:
        st.info("No preview renderer is available for this dashboard page.")


def _render_real_time_preview() -> None:
    txns = demo_transactions()
    volume = (
        txns.assign(time_bucket=txns["transaction_timestamp"].dt.floor("15min"))
        .groupby("time_bucket", as_index=False)
        .size()
        .rename(columns={"size": "txn_count"})
    )
    severity = pd.DataFrame(
        {
            "severity": ["critical", "high", "medium", "low"],
            "count": [6, 11, 18, 25],
        }
    )
    fraud_rate = float((txns["status"] == "flagged").mean() * 100)

    cards = st.columns(4)
    cards[0].markdown(
        kpi_card_html("Transactions", f"{len(txns.index):,}", "12.4%", True, "TRX"),
        unsafe_allow_html=True,
    )
    cards[1].markdown(
        kpi_card_html("Fraud Rate", f"{fraud_rate:.2f}%", "2.1%", False, "RISK"),
        unsafe_allow_html=True,
    )
    cards[2].markdown(
        kpi_card_html("Avg Risk", f"{txns['risk_score'].mean():.3f}", "0.8%", False, "AVG"),
        unsafe_allow_html=True,
    )
    cards[3].markdown(
        kpi_card_html("Active Alerts", "17", "4.0%", False, "ALRT"),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([2, 1])
    c1.plotly_chart(transaction_volume_chart(volume), width="stretch")
    c2.plotly_chart(fraud_rate_gauge(fraud_rate), width="stretch")

    c3, c4 = st.columns(2)
    c3.plotly_chart(risk_score_histogram(txns), width="stretch")
    c4.plotly_chart(alert_severity_pie(severity), width="stretch")
    st.markdown("### Live Transaction Feed")
    st.markdown(live_feed_table_html(txns.head(12)), unsafe_allow_html=True)


def _render_model_performance_preview() -> None:
    scores = demo_model_scores()
    labeled = scores.copy()
    current_auc = roc_auc_score(labeled["actual_label"], labeled["overall_score"])
    current = labeled["overall_score"]
    baseline = pd.Series(np.clip(current.to_numpy() * 0.97 + 0.015, 0.0, 1.0))
    psi = calculate_population_stability_index(baseline, current)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Scored Transactions", f"{len(scores.index):,}")
    k2.metric("Labeled Outcomes", f"{len(labeled.index):,}")
    k3.metric("Current ROC AUC", f"{current_auc:.3f}")
    k4.metric("Score PSI", f"{psi:.4f}")

    matrix = build_confusion_matrix(scores)
    fig_matrix = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        aspect="auto",
    )
    fig_matrix.update_layout(template="plotly_dark", height=360)

    roc_fig = go.Figure()
    pr_fig = go.Figure()
    comparison_rows: list[dict[str, Any]] = []
    for model_version, group in scores.groupby("model_version"):
        fpr, tpr, _ = roc_curve(group["actual_label"], group["overall_score"])
        precision, recall, _ = precision_recall_curve(group["actual_label"], group["overall_score"])
        model_auc = roc_auc_score(group["actual_label"], group["overall_score"])
        average_precision = float(np.trapz(precision, recall))
        roc_fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model_version} AUC {model_auc:.3f}")
        )
        pr_fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{model_version} AP {average_precision:.3f}",
            )
        )
        comparison_rows.append(
            {
                "model_version": model_version,
                "metric": "ROC AUC",
                "value": model_auc,
            }
        )
        comparison_rows.append(
            {
                "model_version": model_version,
                "metric": "Avg Precision",
                "value": average_precision,
            }
        )
    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"dash": "dash", "color": "#95a5a6"},
        )
    )
    roc_fig.update_layout(template="plotly_dark", title="ROC Curves", height=360)
    pr_fig.update_layout(template="plotly_dark", title="Precision-Recall Curves", height=360)
    roc_fig.update_xaxes(title="False Positive Rate", range=[0, 1])
    roc_fig.update_yaxes(title="True Positive Rate", range=[0, 1])
    pr_fig.update_xaxes(title="Recall", range=[0, 1])
    pr_fig.update_yaxes(title="Precision", range=[0, 1])

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_matrix, width="stretch")
    c2.plotly_chart(roc_fig, width="stretch")

    labeled["label"] = np.where(labeled["actual_label"] == 1, "Fraud", "Legitimate")
    score_dist = px.histogram(
        labeled,
        x="overall_score",
        color="label",
        barmode="overlay",
        nbins=20,
        opacity=0.75,
        title="Score Distribution by Actual Label",
        color_discrete_map={"Fraud": "#e74c3c", "Legitimate": "#3498db"},
    )
    score_dist.update_xaxes(range=[0, 1])
    score_dist.update_layout(template="plotly_dark", height=360)

    feature_df = pd.DataFrame(
        {
            "feature": [
                "merchant_risk",
                "velocity_1h",
                "amount_zscore",
                "device_trust",
                "geo_distance",
                "account_age_days",
            ],
            "mean_abs_contribution": [0.31, 0.27, 0.22, 0.18, 0.14, 0.11],
        }
    ).sort_values("mean_abs_contribution")
    feature_fig = px.bar(
        feature_df,
        x="mean_abs_contribution",
        y="feature",
        orientation="h",
        title="Feature Importance",
        color="mean_abs_contribution",
        color_continuous_scale="Teal",
    )
    feature_fig.update_layout(template="plotly_dark", height=360)

    c3, c4 = st.columns(2)
    c3.plotly_chart(pr_fig, width="stretch")
    c4.plotly_chart(score_dist, width="stretch")
    c5, c6 = st.columns(2)
    c5.plotly_chart(feature_fig, width="stretch")
    c6.plotly_chart(
        px.histogram(
            pd.DataFrame(
                {
                    "score": pd.concat([baseline, current], ignore_index=True),
                    "window": ["Baseline"] * len(baseline.index) + ["Current"] * len(current.index),
                }
            ),
            x="score",
            color="window",
            barmode="overlay",
            nbins=20,
            opacity=0.65,
            title=f"Prediction Drift Detection (PSI {psi:.4f})",
        ).update_layout(template="plotly_dark", height=360),
        width="stretch",
    )

    c7, c8 = st.columns(2)
    c7.plotly_chart(
        px.line(
            build_auc_trend(scores, min_labels=10),
            x="period",
            y="auc",
            color="model_version",
            markers=True,
            title="ROC AUC Tracking Over Time",
        ).update_layout(template="plotly_dark", height=360),
        width="stretch",
    )
    c8.plotly_chart(
        px.bar(
            pd.DataFrame(comparison_rows),
            x="model_version",
            y="value",
            color="metric",
            barmode="group",
            title="Model A/B Comparison",
        ).update_layout(template="plotly_dark", height=360),
        width="stretch",
    )

    alerts = build_degradation_alerts(scores, baseline_scores=baseline, current_scores=current)
    pr_summary = build_precision_recall_summary(scores)
    st.dataframe(pr_summary, width="stretch")
    if alerts.empty:
        st.success("No model degradation alerts in preview data.")
    else:
        st.dataframe(alerts, width="stretch")


def _render_alert_management_preview() -> None:
    alerts = demo_alerts()
    volume = (
        alerts.assign(period=alerts["created_at"].dt.floor("D"))
        .groupby(["period", "severity"], as_index=False)
        .size()
        .rename(columns={"size": "alert_count"})
    )
    rule_df = (
        alerts.groupby(["rule_id", "rule_name", "rule_category"], as_index=False)
        .agg(
            triggered_count=("alert_id", "count"),
            confirmed_count=("status", lambda value: int((value == "resolved").sum())),
            false_positive_count=("status", lambda value: int((value == "false_positive").sum())),
            closed_count=(
                "status",
                lambda value: int(value.isin(["resolved", "false_positive"]).sum()),
            ),
            avg_resolution_hours=("resolution_hours", "mean"),
        )
        .pipe(calculate_rule_effectiveness)
    )
    kpis = calculate_alert_kpis(alerts)

    cols = st.columns(6)
    cols[0].metric("Total Alerts", f"{int(kpis['total_alerts']):,}")
    cols[1].metric("Open / Active", f"{int(kpis['open_alerts']):,}")
    cols[2].metric("Resolution Rate", f"{kpis['resolution_rate']:.1%}")
    cols[3].metric("False Positive Rate", f"{kpis['false_positive_rate']:.1%}")
    cols[4].metric("SLA Compliance", f"{kpis['sla_compliance']:.1%}")
    cols[5].metric("Avg Response", f"{kpis['avg_response_hours']:.2f}h")

    c1, c2 = st.columns([2, 1])
    c1.plotly_chart(
        px.area(
            volume,
            x="period",
            y="alert_count",
            color="severity",
            title="Alert Volume by Severity",
        ).update_layout(template="plotly_dark", height=360),
        width="stretch",
    )
    analyst = calculate_resolution_by_analyst(alerts)
    c2.plotly_chart(
        px.bar(
            analyst,
            x="assigned_to",
            y="resolution_rate",
            color="closed_alerts",
            title="Analyst Resolution Rate",
        ).update_layout(template="plotly_dark", height=360),
        width="stretch",
    )

    sla = calculate_sla_metrics(alerts)
    sla_summary = sla.groupby(["severity", "sla_status"]).size().reset_index(name="alerts")
    c3, c4 = st.columns(2)
    c3.plotly_chart(
        px.bar(
            sla_summary,
            x="severity",
            y="alerts",
            color="sla_status",
            title="SLA Compliance by Severity",
        ).update_layout(template="plotly_dark", height=360),
        width="stretch",
    )
    c4.plotly_chart(
        px.scatter(
            rule_df,
            x="triggered_count",
            y="precision_proxy",
            size="closed_count",
            color="rule_category",
            hover_name="rule_name",
            title="Rule Effectiveness",
        ).update_layout(template="plotly_dark", height=360),
        width="stretch",
    )
    st.dataframe(rule_df, width="stretch")


def _render_trend_preview() -> None:
    txns = demo_transactions()
    daily = (
        txns.assign(period=txns["transaction_timestamp"].dt.floor("D"))
        .groupby(["period", "channel"], as_index=False)
        .agg(txn_count=("transaction_id", "count"), avg_risk=("risk_score", "mean"))
    )
    st.plotly_chart(
        px.line(
            daily,
            x="period",
            y="txn_count",
            color="channel",
            markers=True,
            title="Transaction Volume Trend",
        ).update_layout(template="plotly_dark", height=360),
        width="stretch",
    )
    st.plotly_chart(
        px.bar(
            daily,
            x="channel",
            y="avg_risk",
            color="channel",
            title="Average Risk by Channel",
        ).update_layout(template="plotly_dark", height=360),
        width="stretch",
    )


def _render_investigation_preview() -> None:
    alerts = demo_alerts().head(20)
    st.metric("Open Investigations", int(alerts["status"].isin(["open", "investigating"]).sum()))
    st.dataframe(
        alerts[
            [
                "alert_id",
                "severity",
                "status",
                "assigned_to",
                "rule_name",
                "risk_score",
                "created_at",
            ]
        ],
        width="stretch",
    )
