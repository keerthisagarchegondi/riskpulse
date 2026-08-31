"""Alert management analytics page for RiskPulse."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text
from sqlalchemy.engine import Engine

from dashboards.streamlit.components.tables import dataframe_to_csv_bytes

SLA_HOURS_BY_SEVERITY: dict[str, float] = {
    "critical": 2.0,
    "high": 4.0,
    "medium": 24.0,
    "low": 48.0,
}


def _run_query(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Execute SQL and return records as a DataFrame."""
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        rows = result.fetchall()
        cols = list(result.keys())
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _render_filters() -> dict[str, Any]:
    """Render page-level alert analytics filters."""
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        start_date = st.date_input(
            "Start Date",
            value=date.today() - timedelta(days=30),
            key="alert_mgmt_start",
        )
    with c2:
        end_date = st.date_input("End Date", value=date.today(), key="alert_mgmt_end")
    with c3:
        grain = st.selectbox("Trend Grain", options=["hour", "day", "week"], index=1)
    with c4:
        severity = st.selectbox(
            "Severity",
            options=["all", "critical", "high", "medium", "low"],
            index=0,
        )

    return {
        "start_ts": datetime.combine(start_date, time.min, tzinfo=timezone.utc),
        "end_ts": datetime.combine(end_date, time.max, tzinfo=timezone.utc),
        "grain": grain,
        "severity": None if severity == "all" else severity,
    }


def _severity_clause(filters: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params: dict[str, Any] = {
        "start_ts": filters["start_ts"],
        "end_ts": filters["end_ts"],
        "grain": filters["grain"],
    }
    clause = "fa.created_at BETWEEN :start_ts AND :end_ts"
    if filters.get("severity"):
        clause += " AND fa.severity = :severity"
        params["severity"] = filters["severity"]
    return clause, params


def _fetch_alert_lifecycle(engine: Engine, filters: dict[str, Any]) -> pd.DataFrame:
    """Fetch alert lifecycle rows for management analytics."""
    where_sql, params = _severity_clause(filters)
    sql = f"""
        SELECT
            fa.alert_id,
            fa.alert_type,
            fa.rule_id,
            COALESCE(fr.rule_name, 'Unknown Rule') AS rule_name,
            COALESCE(fr.rule_category, 'unknown') AS rule_category,
            fa.severity,
            fa.status,
            fa.assigned_to,
            fa.created_at,
            fa.updated_at,
            fa.resolved_at,
            COALESCE(fa.risk_score, rs.overall_score, 0)::FLOAT AS risk_score,
            EXTRACT(EPOCH FROM (fa.updated_at - fa.created_at)) / 3600 AS response_hours,
            EXTRACT(EPOCH FROM (fa.resolved_at - fa.created_at)) / 3600 AS resolution_hours
        FROM fraud_alerts fa
        LEFT JOIN fraud_rules fr ON fr.rule_id = fa.rule_id
        LEFT JOIN LATERAL (
            SELECT rs2.overall_score
            FROM risk_scores rs2
            WHERE rs2.transaction_id = fa.transaction_id
            ORDER BY rs2.scoring_timestamp DESC
            LIMIT 1
        ) rs ON TRUE
        WHERE {where_sql}
        ORDER BY fa.created_at DESC
        LIMIT 100000
    """  # nosec B608
    df = _run_query(engine, sql, params)
    if not df.empty:
        for col in ["created_at", "updated_at", "resolved_at"]:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        for col in ["response_hours", "resolution_hours", "risk_score"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fetch_alert_volume(engine: Engine, filters: dict[str, Any]) -> pd.DataFrame:
    """Fetch alert volume grouped by selected grain and severity."""
    where_sql, params = _severity_clause(filters)
    sql = f"""
        SELECT
            date_trunc(:grain, fa.created_at) AS period,
            fa.severity,
            COUNT(*) AS alert_count
        FROM fraud_alerts fa
        WHERE {where_sql}
        GROUP BY 1, 2
        ORDER BY 1, 2
    """  # nosec B608
    return _run_query(engine, sql, params)


def _fetch_rule_effectiveness(engine: Engine, filters: dict[str, Any]) -> pd.DataFrame:
    """Fetch alert rule effectiveness metrics."""
    where_sql, params = _severity_clause(filters)
    sql = f"""
        SELECT
            COALESCE(fa.rule_id, 'unassigned') AS rule_id,
            COALESCE(fr.rule_name, 'Unassigned / Model Alert') AS rule_name,
            COALESCE(fr.rule_category, 'model') AS rule_category,
            COUNT(*) AS triggered_count,
            COUNT(*) FILTER (WHERE fa.status = 'resolved') AS confirmed_count,
            COUNT(*) FILTER (WHERE fa.status = 'false_positive') AS false_positive_count,
            COUNT(*) FILTER (WHERE fa.status IN ('resolved', 'false_positive')) AS closed_count,
            COALESCE(
                AVG(EXTRACT(EPOCH FROM (fa.resolved_at - fa.created_at)) / 3600)
                FILTER (WHERE fa.resolved_at IS NOT NULL),
                0
            ) AS avg_resolution_hours
        FROM fraud_alerts fa
        LEFT JOIN fraud_rules fr ON fr.rule_id = fa.rule_id
        WHERE {where_sql}
        GROUP BY 1, 2, 3
        ORDER BY triggered_count DESC
        LIMIT 100
    """  # nosec B608
    df = _run_query(engine, sql, params)
    if df.empty:
        return df
    return calculate_rule_effectiveness(df)


def calculate_sla_metrics(
    alerts: pd.DataFrame,
    sla_hours_by_severity: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Calculate SLA target, elapsed hours, and compliance per alert."""
    sla_hours = sla_hours_by_severity or SLA_HOURS_BY_SEVERITY
    if alerts.empty:
        return alerts.copy()

    df = alerts.copy()
    now = pd.Timestamp.now(tz=timezone.utc)
    created = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    resolved = pd.to_datetime(df.get("resolved_at"), utc=True, errors="coerce")
    effective_end = resolved.fillna(now)

    df["sla_target_hours"] = df["severity"].map(sla_hours).fillna(24.0).astype(float)
    df["elapsed_hours"] = (effective_end - created).dt.total_seconds() / 3600
    df["sla_met"] = df["elapsed_hours"] <= df["sla_target_hours"]
    df["sla_status"] = np.select(
        [
            df["sla_met"],
            df["status"].isin(["open", "investigating"]) & ~df["sla_met"],
        ],
        ["met", "breached_open"],
        default="breached_closed",
    )
    return df


def calculate_alert_kpis(alerts: pd.DataFrame) -> dict[str, float]:
    """Calculate top-level alert operations KPIs."""
    if alerts.empty:
        return {
            "total_alerts": 0,
            "open_alerts": 0,
            "resolution_rate": 0.0,
            "false_positive_rate": 0.0,
            "sla_compliance": 0.0,
            "avg_response_hours": 0.0,
        }

    sla_df = calculate_sla_metrics(alerts)
    closed_mask = alerts["status"].isin(["resolved", "false_positive"])
    closed_count = int(closed_mask.sum())
    false_positive_count = int((alerts["status"] == "false_positive").sum())
    response_hours = pd.to_numeric(alerts.get("response_hours"), errors="coerce").dropna()

    return {
        "total_alerts": float(len(alerts.index)),
        "open_alerts": float((alerts["status"].isin(["open", "investigating"])).sum()),
        "resolution_rate": closed_count / len(alerts.index),
        "false_positive_rate": false_positive_count / closed_count if closed_count else 0.0,
        "sla_compliance": float(sla_df["sla_met"].mean()) if not sla_df.empty else 0.0,
        "avg_response_hours": float(response_hours.mean()) if not response_hours.empty else 0.0,
    }


def calculate_resolution_by_analyst(alerts: pd.DataFrame) -> pd.DataFrame:
    """Calculate analyst resolution rate and throughput."""
    if alerts.empty:
        return pd.DataFrame(
            columns=[
                "assigned_to",
                "assigned_alerts",
                "closed_alerts",
                "resolution_rate",
                "false_positive_rate",
                "avg_resolution_hours",
            ]
        )

    df = alerts.copy()
    df["assigned_to"] = df["assigned_to"].fillna("unassigned")
    grouped = df.groupby("assigned_to", dropna=False)
    rows: list[dict[str, Any]] = []
    for analyst, group in grouped:
        closed = group[group["status"].isin(["resolved", "false_positive"])]
        false_positive = group[group["status"] == "false_positive"]
        resolution_hours = pd.to_numeric(closed.get("resolution_hours"), errors="coerce").dropna()
        rows.append(
            {
                "assigned_to": analyst,
                "assigned_alerts": int(len(group.index)),
                "closed_alerts": int(len(closed.index)),
                "resolution_rate": (
                    len(closed.index) / len(group.index) if len(group.index) else 0.0
                ),
                "false_positive_rate": (
                    len(false_positive.index) / len(closed.index) if len(closed.index) else 0.0
                ),
                "avg_resolution_hours": (
                    float(resolution_hours.mean()) if not resolution_hours.empty else 0.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("assigned_alerts", ascending=False)


def calculate_rule_effectiveness(rule_df: pd.DataFrame) -> pd.DataFrame:
    """Add precision and false positive rate columns to rule metrics."""
    if rule_df.empty:
        return rule_df.copy()
    df = rule_df.copy()
    closed = pd.to_numeric(df["closed_count"], errors="coerce").fillna(0)
    confirmed = pd.to_numeric(df["confirmed_count"], errors="coerce").fillna(0)
    false_positive = pd.to_numeric(df["false_positive_count"], errors="coerce").fillna(0)
    df["precision_proxy"] = np.where(closed > 0, confirmed / closed, 0.0)
    df["false_positive_rate"] = np.where(closed > 0, false_positive / closed, 0.0)
    df["action_hint"] = np.select(
        [
            (df["triggered_count"] >= 10) & (df["false_positive_rate"] >= 0.5),
            (df["triggered_count"] >= 10) & (df["precision_proxy"] >= 0.8),
        ],
        ["review_threshold", "high_value"],
        default="monitor",
    )
    return df


def _render_kpis(alerts: pd.DataFrame) -> None:
    kpis = calculate_alert_kpis(alerts)
    cols = st.columns(6)
    cols[0].metric("Total Alerts", f"{int(kpis['total_alerts']):,}")
    cols[1].metric("Open / Active", f"{int(kpis['open_alerts']):,}")
    cols[2].metric("Resolution Rate", f"{kpis['resolution_rate']:.1%}")
    cols[3].metric("False Positive Rate", f"{kpis['false_positive_rate']:.1%}")
    cols[4].metric("SLA Compliance", f"{kpis['sla_compliance']:.1%}")
    cols[5].metric("Avg Response", f"{kpis['avg_response_hours']:.2f}h")


def _render_alert_volume(volume_df: pd.DataFrame) -> None:
    st.markdown("### Alert Volume")
    if volume_df.empty:
        st.info("No alert volume data available.")
        return
    fig = px.area(
        volume_df,
        x="period",
        y="alert_count",
        color="severity",
        title="Alert Volume by Severity",
        color_discrete_map={
            "low": "#2ecc71",
            "medium": "#f39c12",
            "high": "#e74c3c",
            "critical": "#8e44ad",
        },
    )
    fig.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig, width="stretch")


def _render_response_metrics(alerts: pd.DataFrame) -> None:
    st.markdown("### Response Time Metrics")
    if alerts.empty:
        st.info("No response metrics available.")
        return

    c1, c2 = st.columns(2)
    response = alerts.dropna(subset=["response_hours"])
    resolution = alerts.dropna(subset=["resolution_hours"])

    with c1:
        fig = px.box(
            response,
            x="severity",
            y="response_hours",
            points="outliers",
            title="Response Hours by Severity",
        )
        fig.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.box(
            resolution,
            x="severity",
            y="resolution_hours",
            points="outliers",
            title="Resolution Hours by Severity",
        )
        fig.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig, width="stretch")


def _render_resolution_by_analyst(alerts: pd.DataFrame) -> None:
    st.markdown("### Resolution Rate by Analyst")
    analyst_df = calculate_resolution_by_analyst(alerts)
    if analyst_df.empty:
        st.info("No analyst assignment data available.")
        return

    fig = px.bar(
        analyst_df,
        x="assigned_to",
        y="resolution_rate",
        color="closed_alerts",
        title="Analyst Resolution Rate",
        color_continuous_scale="Viridis",
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(template="plotly_dark", height=360)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(analyst_df, width="stretch")


def _render_false_positive_tracking(alerts: pd.DataFrame) -> None:
    st.markdown("### False Positive Rate Tracking")
    if alerts.empty:
        st.info("No false positive tracking data available.")
        return

    df = alerts.copy()
    df["period"] = df["created_at"].dt.floor("D")
    grouped = (
        df.groupby("period")
        .agg(
            closed=("status", lambda s: int(s.isin(["resolved", "false_positive"]).sum())),
            false_positives=("status", lambda s: int((s == "false_positive").sum())),
        )
        .reset_index()
    )
    grouped["false_positive_rate"] = np.where(
        grouped["closed"] > 0,
        grouped["false_positives"] / grouped["closed"],
        0.0,
    )
    fig = px.line(
        grouped,
        x="period",
        y="false_positive_rate",
        markers=True,
        title="False Positive Rate Over Time",
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(template="plotly_dark", height=360)
    st.plotly_chart(fig, width="stretch")


def _render_sla_compliance(alerts: pd.DataFrame) -> None:
    st.markdown("### SLA Compliance Monitoring")
    sla_df = calculate_sla_metrics(alerts)
    if sla_df.empty:
        st.info("No SLA data available.")
        return

    summary = (
        sla_df.groupby(["severity", "sla_status"])
        .size()
        .reset_index(name="alerts")
        .sort_values(["severity", "sla_status"])
    )
    fig = px.bar(
        summary,
        x="severity",
        y="alerts",
        color="sla_status",
        barmode="stack",
        title="SLA Compliance by Severity",
        color_discrete_map={
            "met": "#2ecc71",
            "breached_open": "#e74c3c",
            "breached_closed": "#f39c12",
        },
    )
    fig.update_layout(template="plotly_dark", height=360)
    st.plotly_chart(fig, width="stretch")

    breached = sla_df.loc[
        ~sla_df["sla_met"],
        [
            "alert_id",
            "severity",
            "status",
            "assigned_to",
            "elapsed_hours",
            "sla_target_hours",
        ],
    ].sort_values("elapsed_hours", ascending=False)
    if not breached.empty:
        st.dataframe(breached.head(100), width="stretch")


def _render_rule_effectiveness(rule_df: pd.DataFrame) -> None:
    st.markdown("### Alert Rule Effectiveness")
    if rule_df.empty:
        st.info("No rule effectiveness data available.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.scatter(
            rule_df,
            x="triggered_count",
            y="precision_proxy",
            size="closed_count",
            color="rule_category",
            hover_name="rule_name",
            title="Rule Precision vs Trigger Volume",
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure(
            go.Bar(
                x=rule_df.head(10)["false_positive_rate"],
                y=rule_df.head(10)["rule_name"],
                orientation="h",
                marker_color="#e74c3c",
            )
        )
        fig.update_layout(
            title="Top Rule False Positive Rates",
            template="plotly_dark",
            height=420,
            yaxis={"autorange": "reversed"},
        )
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    st.dataframe(rule_df, width="stretch")


def render(engine: Engine) -> None:
    """Render alert management analytics dashboard."""
    st.markdown("## Alert Management Analytics")
    st.caption("Monitor alert load, analyst throughput, SLA health, and rule quality.")

    filters = _render_filters()
    with st.spinner("Loading alert analytics..."):
        alerts = _fetch_alert_lifecycle(engine, filters)
        volume_df = _fetch_alert_volume(engine, filters)
        rule_df = _fetch_rule_effectiveness(engine, filters)

    _render_kpis(alerts)
    st.markdown("---")

    tab_volume, tab_response, tab_analysts, tab_sla, tab_rules = st.tabs(
        [
            "Volume",
            "Response",
            "Analysts",
            "SLA",
            "Rules",
        ]
    )

    with tab_volume:
        _render_alert_volume(volume_df)
        _render_false_positive_tracking(alerts)

    with tab_response:
        _render_response_metrics(alerts)

    with tab_analysts:
        _render_resolution_by_analyst(alerts)

    with tab_sla:
        _render_sla_compliance(alerts)

    with tab_rules:
        _render_rule_effectiveness(rule_df)
        if not rule_df.empty:
            st.download_button(
                "Export Rule Effectiveness (CSV)",
                data=dataframe_to_csv_bytes(rule_df),
                file_name=f"alert_rule_effectiveness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    st.caption(f"Last refreshed: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
