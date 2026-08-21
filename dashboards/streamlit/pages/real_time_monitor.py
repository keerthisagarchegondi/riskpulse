"""Real-Time Monitoring page for the RiskPulse Streamlit dashboard.

Renders a live-updating view of transaction activity, fraud metrics,
alert severity breakdowns, and geographic distribution.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text
from sqlalchemy.engine import Engine

from dashboards.streamlit.components.charts import (
    alert_severity_pie,
    channel_breakdown_bar,
    fraud_rate_gauge,
    geo_heatmap,
    kpi_card_html,
    live_feed_table_html,
    risk_score_histogram,
    transaction_volume_chart,
)
from dashboards.streamlit.components.filters import render_sidebar_filters

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data access helpers (sync psycopg2 queries)
# ---------------------------------------------------------------------------


def _run_query(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Execute a read-only SQL query and return results as a DataFrame."""
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        rows = result.fetchall()
        columns = list(result.keys())
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def _fetch_kpis(engine: Engine, filters: dict[str, Any]) -> dict[str, Any]:
    """Fetch KPI aggregates for the selected time window."""
    sql = """
        SELECT
            COUNT(*)                                                  AS total_txns,
            COUNT(*) FILTER (WHERE t.status = 'flagged')              AS flagged_txns,
            COALESCE(AVG(rs.overall_score), 0)                        AS avg_risk_score,
            COUNT(DISTINCT fa.alert_id) FILTER (WHERE fa.status = 'open') AS active_alerts
        FROM transactions t
        LEFT JOIN risk_scores rs ON rs.transaction_id = t.transaction_id
        LEFT JOIN fraud_alerts fa ON fa.transaction_id = t.transaction_id
        WHERE t.transaction_timestamp BETWEEN :time_start AND :time_end
    """
    params: dict[str, Any] = {"time_start": filters["time_start"], "time_end": filters["time_end"]}

    cond, params = _apply_optional_filters(filters, params)
    sql += cond

    df = _run_query(engine, sql, params)
    if df.empty:
        return {"total_txns": 0, "flagged_txns": 0, "avg_risk_score": 0.0, "active_alerts": 0}
    row = df.iloc[0]
    return {
        "total_txns": int(row["total_txns"]),
        "flagged_txns": int(row["flagged_txns"]),
        "avg_risk_score": float(row["avg_risk_score"]),
        "active_alerts": int(row["active_alerts"]),
    }


def _fetch_previous_kpis(engine: Engine, filters: dict[str, Any]) -> dict[str, Any]:
    """Fetch KPIs for the previous equivalent time window (for delta)."""
    window = filters["time_end"] - filters["time_start"]
    prev_filters = {
        **filters,
        "time_start": filters["time_start"] - window,
        "time_end": filters["time_start"],
    }
    return _fetch_kpis(engine, prev_filters)


def _fetch_transaction_volume(engine: Engine, filters: dict[str, Any]) -> pd.DataFrame:
    """Transaction count grouped by time buckets."""
    sql = """
        SELECT
            date_trunc('minute', t.transaction_timestamp) AS time_bucket,
            COUNT(*)                                      AS txn_count
        FROM transactions t
        WHERE t.transaction_timestamp BETWEEN :time_start AND :time_end
    """
    params: dict[str, Any] = {"time_start": filters["time_start"], "time_end": filters["time_end"]}
    cond, params = _apply_optional_filters(filters, params)
    sql += cond
    sql += " GROUP BY 1 ORDER BY 1"
    return _run_query(engine, sql, params)


def _fetch_risk_scores(engine: Engine, filters: dict[str, Any]) -> pd.DataFrame:
    """Fetch overall risk scores for histogram."""
    sql = """
        SELECT rs.overall_score
        FROM risk_scores rs
        JOIN transactions t ON t.transaction_id = rs.transaction_id
        WHERE t.transaction_timestamp BETWEEN :time_start AND :time_end
    """
    params: dict[str, Any] = {"time_start": filters["time_start"], "time_end": filters["time_end"]}
    cond, params = _apply_optional_filters(filters, params)
    sql += cond
    return _run_query(engine, sql, params)


def _fetch_alert_severity(engine: Engine, filters: dict[str, Any]) -> pd.DataFrame:
    """Alert count by severity."""
    sql = """
        SELECT fa.severity, COUNT(*) AS count
        FROM fraud_alerts fa
        JOIN transactions t ON t.transaction_id = fa.transaction_id
        WHERE t.transaction_timestamp BETWEEN :time_start AND :time_end
    """
    params: dict[str, Any] = {"time_start": filters["time_start"], "time_end": filters["time_end"]}
    if filters.get("severity"):
        sql += " AND fa.severity = :severity"
        params["severity"] = filters["severity"]
    sql += " GROUP BY fa.severity ORDER BY fa.severity"
    return _run_query(engine, sql, params)


def _fetch_geo_distribution(engine: Engine, filters: dict[str, Any]) -> pd.DataFrame:
    """Transaction count by country."""
    sql = """
        SELECT t.geo_country, COUNT(*) AS txn_count
        FROM transactions t
        WHERE t.transaction_timestamp BETWEEN :time_start AND :time_end
          AND t.geo_country IS NOT NULL
    """
    params: dict[str, Any] = {"time_start": filters["time_start"], "time_end": filters["time_end"]}
    cond, params = _apply_optional_filters(filters, params)
    sql += cond
    sql += " GROUP BY t.geo_country ORDER BY txn_count DESC LIMIT 50"
    return _run_query(engine, sql, params)


def _fetch_channel_breakdown(engine: Engine, filters: dict[str, Any]) -> pd.DataFrame:
    """Transaction and fraud counts by channel."""
    sql = """
        SELECT
            t.channel,
            COUNT(*)                                       AS txn_count,
            COUNT(*) FILTER (WHERE t.status = 'flagged')   AS fraud_count
        FROM transactions t
        WHERE t.transaction_timestamp BETWEEN :time_start AND :time_end
    """
    params: dict[str, Any] = {"time_start": filters["time_start"], "time_end": filters["time_end"]}
    cond, params = _apply_optional_filters(filters, params)
    sql += cond
    sql += " GROUP BY t.channel ORDER BY txn_count DESC"
    return _run_query(engine, sql, params)


def _fetch_live_feed(engine: Engine, filters: dict[str, Any], limit: int = 100) -> pd.DataFrame:
    """Most recent transactions with risk score."""
    sql = """
        SELECT
            t.transaction_id,
            t.transaction_amount,
            t.status,
            COALESCE(rs.overall_score, 0) AS risk_score,
            t.channel,
            t.geo_country,
            t.transaction_timestamp
        FROM transactions t
        LEFT JOIN LATERAL (
            SELECT rs2.overall_score
            FROM risk_scores rs2
            WHERE rs2.transaction_id = t.transaction_id
            ORDER BY rs2.scoring_timestamp DESC
            LIMIT 1
        ) rs ON TRUE
        WHERE t.transaction_timestamp BETWEEN :time_start AND :time_end
    """
    params: dict[str, Any] = {"time_start": filters["time_start"], "time_end": filters["time_end"]}
    cond, params = _apply_optional_filters(filters, params)
    sql += cond
    sql += " ORDER BY t.transaction_timestamp DESC LIMIT :feed_limit"
    params["feed_limit"] = limit
    return _run_query(engine, sql, params)


def _apply_optional_filters(
    filters: dict[str, Any],
    params: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Build additional WHERE clause fragments for optional filters."""
    clauses: list[str] = []
    if filters.get("status"):
        clauses.append(" AND t.status = :filter_status")
        params["filter_status"] = filters["status"]
    if filters.get("channel"):
        clauses.append(" AND t.channel = :filter_channel")
        params["filter_channel"] = filters["channel"]
    if filters.get("risk_min") is not None and filters.get("risk_max") is not None:
        risk_min = filters["risk_min"]
        risk_max = filters["risk_max"]
        if not (risk_min == 0.0 and risk_max == 1.0):
            clauses.append(
                " AND EXISTS ("
                "SELECT 1 FROM risk_scores rs_f "
                "WHERE rs_f.transaction_id = t.transaction_id "
                "AND rs_f.overall_score BETWEEN :risk_min AND :risk_max)"
            )
            params["risk_min"] = risk_min
            params["risk_max"] = risk_max
    return "".join(clauses), params


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------


def _compute_delta(current: float, previous: float) -> tuple[str, bool]:
    """Return a human-readable delta string and whether it's positive."""
    if previous == 0:
        return "N/A", True
    pct = ((current - previous) / previous) * 100
    return f"{abs(pct):.1f}%", pct >= 0


def render(engine: Engine) -> None:
    """Render the Real-Time Monitoring page."""
    st.markdown("## 📡 Real-Time Monitoring")

    filters = render_sidebar_filters(key_prefix="rtm")

    # --- KPI row ---
    with st.spinner("Loading metrics…"):
        kpis = _fetch_kpis(engine, filters)
        prev_kpis = _fetch_previous_kpis(engine, filters)

    total_delta, total_pos = _compute_delta(kpis["total_txns"], prev_kpis["total_txns"])
    fraud_rate = (kpis["flagged_txns"] / kpis["total_txns"] * 100) if kpis["total_txns"] else 0.0
    prev_fraud_rate = (
        (prev_kpis["flagged_txns"] / prev_kpis["total_txns"] * 100)
        if prev_kpis["total_txns"]
        else 0.0
    )
    fraud_delta, fraud_pos = _compute_delta(fraud_rate, prev_fraud_rate)
    score_delta, score_pos = _compute_delta(kpis["avg_risk_score"], prev_kpis["avg_risk_score"])
    alert_delta, alert_pos = _compute_delta(kpis["active_alerts"], prev_kpis["active_alerts"])

    kpi_cols = st.columns(4)
    kpi_data = [
        ("Total Transactions", f"{kpis['total_txns']:,}", total_delta, total_pos, "📈"),
        ("Fraud Rate", f"{fraud_rate:.2f}%", fraud_delta, not fraud_pos, "🚨"),
        ("Avg Risk Score", f"{kpis['avg_risk_score']:.4f}", score_delta, not score_pos, "🎯"),
        ("Active Alerts", f"{kpis['active_alerts']:,}", alert_delta, not alert_pos, "🔔"),
    ]
    for col, (title, value, delta, positive, icon) in zip(kpi_cols, kpi_data):
        col.markdown(kpi_card_html(title, value, delta, positive, icon), unsafe_allow_html=True)

    st.markdown("---")

    # --- Row 1: Volume + Fraud Gauge ---
    col_vol, col_gauge = st.columns([2, 1])

    with col_vol:
        vol_df = _fetch_transaction_volume(engine, filters)
        if not vol_df.empty:
            st.plotly_chart(transaction_volume_chart(vol_df), use_container_width=True)
        else:
            st.info("No transaction volume data for the selected window.")

    with col_gauge:
        st.plotly_chart(fraud_rate_gauge(fraud_rate), use_container_width=True)

    # --- Row 2: Risk Histogram + Alert Severity ---
    col_hist, col_pie = st.columns(2)

    with col_hist:
        risk_df = _fetch_risk_scores(engine, filters)
        if not risk_df.empty:
            st.plotly_chart(risk_score_histogram(risk_df), use_container_width=True)
        else:
            st.info("No risk score data available.")

    with col_pie:
        sev_df = _fetch_alert_severity(engine, filters)
        if not sev_df.empty:
            st.plotly_chart(alert_severity_pie(sev_df), use_container_width=True)
        else:
            st.info("No alert data available.")

    # --- Row 3: Geo Heatmap + Channel Breakdown ---
    col_geo, col_chan = st.columns(2)

    with col_geo:
        geo_df = _fetch_geo_distribution(engine, filters)
        if not geo_df.empty:
            st.plotly_chart(geo_heatmap(geo_df), use_container_width=True)
        else:
            st.info("No geographic data available.")

    with col_chan:
        chan_df = _fetch_channel_breakdown(engine, filters)
        if not chan_df.empty:
            st.plotly_chart(channel_breakdown_bar(chan_df), use_container_width=True)
        else:
            st.info("No channel data available.")

    # --- Row 4: Live Feed ---
    st.markdown("### 📋 Live Transaction Feed")
    feed_df = _fetch_live_feed(engine, filters)
    st.markdown(live_feed_table_html(feed_df), unsafe_allow_html=True)

    # Footer
    st.caption(
        f"Last refreshed: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} "
        f"| Window: {filters['time_start'].strftime('%H:%M')}–{filters['time_end'].strftime('%H:%M')} UTC"
    )
