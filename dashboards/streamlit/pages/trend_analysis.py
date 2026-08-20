"""Trend Analysis page for fraud trends, model performance, and rule effectiveness."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text
from sqlalchemy.engine import Engine

from dashboards.streamlit.components.tables import dataframe_to_csv_bytes

_PERIOD_GRAIN = {
    "Daily": "day",
    "Weekly": "week",
    "Monthly": "month",
}


def _run_query(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Execute SQL and return records as DataFrame."""
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        rows = result.fetchall()
        cols = list(result.keys())
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _render_filters() -> dict[str, Any]:
    """Render trend page filters and return normalized values."""
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=90))
    with c2:
        end_date = st.date_input("End Date", value=date.today())
    with c3:
        grain_label = st.selectbox("Aggregation", options=list(_PERIOD_GRAIN.keys()), index=0)

    return {
        "start_ts": datetime.combine(start_date, time.min, tzinfo=timezone.utc),
        "end_ts": datetime.combine(end_date, time.max, tzinfo=timezone.utc),
        "grain": _PERIOD_GRAIN[grain_label],
        "grain_label": grain_label,
    }


def _fetch_fraud_trends(engine: Engine, filters: dict[str, Any]) -> pd.DataFrame:
    """Fraud trend counts over selected time grain."""
    sql = """
        SELECT
            date_trunc(:grain, fa.created_at) AS period,
            COUNT(*) AS total_alerts,
            COUNT(*) FILTER (WHERE fa.status = 'resolved') AS confirmed_fraud,
            COUNT(*) FILTER (WHERE fa.status = 'false_positive') AS false_positives,
            COUNT(*) FILTER (WHERE fa.severity IN ('high', 'critical')) AS high_severity
        FROM fraud_alerts fa
        WHERE fa.created_at BETWEEN :start_ts AND :end_ts
        GROUP BY 1
        ORDER BY 1
    """
    return _run_query(engine, sql, filters)


def _fetch_category_breakdown(
    engine: Engine, filters: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fraud category breakdown across merchant, channel, and geography."""
    merchant_sql = """
        SELECT
            COALESCE(t.merchant_name, 'Unknown') AS merchant,
            COUNT(*) AS alerts,
            COUNT(*) FILTER (WHERE fa.status = 'resolved') AS confirmed_fraud
        FROM fraud_alerts fa
        JOIN transactions t ON t.transaction_id = fa.transaction_id
        WHERE fa.created_at BETWEEN :start_ts AND :end_ts
        GROUP BY 1
        ORDER BY alerts DESC
        LIMIT 20
    """
    channel_sql = """
        SELECT
            t.channel,
            COUNT(*) AS alerts,
            COUNT(*) FILTER (WHERE fa.status = 'resolved') AS confirmed_fraud
        FROM fraud_alerts fa
        JOIN transactions t ON t.transaction_id = fa.transaction_id
        WHERE fa.created_at BETWEEN :start_ts AND :end_ts
        GROUP BY 1
        ORDER BY alerts DESC
    """
    geography_sql = """
        SELECT
            COALESCE(t.geo_country, 'UNK') AS geo_country,
            COUNT(*) AS alerts,
            COUNT(*) FILTER (WHERE fa.status = 'resolved') AS confirmed_fraud
        FROM fraud_alerts fa
        JOIN transactions t ON t.transaction_id = fa.transaction_id
        WHERE fa.created_at BETWEEN :start_ts AND :end_ts
        GROUP BY 1
        ORDER BY alerts DESC
        LIMIT 30
    """
    merchant = _run_query(engine, merchant_sql, filters)
    channel = _run_query(engine, channel_sql, filters)
    geography = _run_query(engine, geography_sql, filters)
    return merchant, channel, geography


def _fetch_model_performance(engine: Engine, filters: dict[str, Any]) -> pd.DataFrame:
    """Model performance over time using analyst-resolved outcomes as labels."""
    sql = """
        SELECT
            date_trunc(:grain, rs.scoring_timestamp) AS period,
            rs.model_version,
            COUNT(*) AS scored_txns,
            AVG(rs.overall_score) AS avg_score,
            COUNT(*) FILTER (WHERE fa.status = 'resolved') AS true_positives,
            COUNT(*) FILTER (WHERE fa.status = 'false_positive') AS false_positives,
            CASE
                WHEN COUNT(*) FILTER (WHERE fa.status IN ('resolved', 'false_positive')) = 0 THEN 0
                ELSE (
                    COUNT(*) FILTER (WHERE fa.status = 'resolved')::DECIMAL
                    / COUNT(*) FILTER (WHERE fa.status IN ('resolved', 'false_positive'))
                )
            END AS precision_proxy
        FROM risk_scores rs
        LEFT JOIN fraud_alerts fa ON fa.transaction_id = rs.transaction_id
        WHERE rs.scoring_timestamp BETWEEN :start_ts AND :end_ts
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    return _run_query(engine, sql, filters)


def _fetch_rule_effectiveness(engine: Engine, filters: dict[str, Any]) -> pd.DataFrame:
    """Rule-level effectiveness metrics over selected date window."""
    sql = """
        SELECT
            fa.rule_id,
            COALESCE(fr.rule_name, 'Unknown Rule') AS rule_name,
            COALESCE(fr.rule_category, 'unknown') AS rule_category,
            COUNT(*) AS triggered_count,
            COUNT(*) FILTER (WHERE fa.status = 'resolved') AS confirmed_fraud,
            COUNT(*) FILTER (WHERE fa.status = 'false_positive') AS false_positives,
            CASE
                WHEN COUNT(*) FILTER (WHERE fa.status IN ('resolved', 'false_positive')) = 0 THEN 0
                ELSE (
                    COUNT(*) FILTER (WHERE fa.status = 'resolved')::DECIMAL
                    / COUNT(*) FILTER (WHERE fa.status IN ('resolved', 'false_positive'))
                )
            END AS precision_proxy
        FROM fraud_alerts fa
        LEFT JOIN fraud_rules fr ON fr.rule_id = fa.rule_id
        WHERE fa.created_at BETWEEN :start_ts AND :end_ts
          AND fa.rule_id IS NOT NULL
        GROUP BY 1, 2, 3
        HAVING COUNT(*) >= 3
        ORDER BY triggered_count DESC
        LIMIT 100
    """
    return _run_query(engine, sql, filters)


def _render_fraud_trends(trends: pd.DataFrame, grain_label: str) -> None:
    """Render trend line chart and KPIs."""
    st.markdown("### Fraud Trends Over Time")
    if trends.empty:
        st.info("No trend data available for selected filters.")
        return

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Alerts", f"{int(trends['total_alerts'].sum()):,}")
    kpi_cols[1].metric("Confirmed Fraud", f"{int(trends['confirmed_fraud'].sum()):,}")
    kpi_cols[2].metric("False Positives", f"{int(trends['false_positives'].sum()):,}")
    kpi_cols[3].metric("High/Critical", f"{int(trends['high_severity'].sum()):,}")

    chart_df = trends.melt(
        id_vars=["period"],
        value_vars=["total_alerts", "confirmed_fraud", "false_positives", "high_severity"],
        var_name="metric",
        value_name="count",
    )
    fig = px.line(
        chart_df,
        x="period",
        y="count",
        color="metric",
        markers=True,
        title=f"Alert Trends ({grain_label})",
        color_discrete_map={
            "total_alerts": "#3498db",
            "confirmed_fraud": "#e74c3c",
            "false_positives": "#95a5a6",
            "high_severity": "#8e44ad",
        },
    )
    fig.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Export Trend Data (CSV)",
        data=dataframe_to_csv_bytes(trends),
        file_name=f"fraud_trends_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def _render_category_charts(
    merchant_df: pd.DataFrame,
    channel_df: pd.DataFrame,
    geography_df: pd.DataFrame,
) -> None:
    """Render category-level distribution charts."""
    st.markdown("### Fraud by Category")
    tab_merchant, tab_channel, tab_geo = st.tabs(["Merchant", "Channel", "Geography"])

    with tab_merchant:
        if merchant_df.empty:
            st.info("No merchant breakdown data available.")
        else:
            fig = px.bar(
                merchant_df.sort_values("alerts", ascending=True),
                x="alerts",
                y="merchant",
                orientation="h",
                color="confirmed_fraud",
                color_continuous_scale="Reds",
                title="Top Merchants by Alert Volume",
            )
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(merchant_df, use_container_width=True)

    with tab_channel:
        if channel_df.empty:
            st.info("No channel data available.")
        else:
            fig = px.bar(
                channel_df,
                x="channel",
                y=["alerts", "confirmed_fraud"],
                barmode="group",
                title="Alerts by Channel",
                color_discrete_sequence=["#3498db", "#e74c3c"],
            )
            fig.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(channel_df, use_container_width=True)

    with tab_geo:
        if geography_df.empty:
            st.info("No geography data available.")
        else:
            fig = px.choropleth(
                geography_df,
                locations="geo_country",
                color="alerts",
                locationmode="ISO-3",
                hover_data=["confirmed_fraud"],
                color_continuous_scale="YlOrRd",
                title="Fraud Alerts by Country",
            )
            fig.update_layout(template="plotly_dark", height=430)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(geography_df, use_container_width=True)


def _render_model_performance(model_df: pd.DataFrame, grain_label: str) -> None:
    """Render model scoring and precision trend charts."""
    st.markdown("### Model Performance Over Time")
    if model_df.empty:
        st.info("No model performance data available.")
        return

    score_fig = px.line(
        model_df,
        x="period",
        y="avg_score",
        color="model_version",
        markers=True,
        title=f"Average Model Risk Score ({grain_label})",
    )
    score_fig.update_layout(template="plotly_dark", height=360)
    st.plotly_chart(score_fig, use_container_width=True)

    precision_fig = px.line(
        model_df,
        x="period",
        y="precision_proxy",
        color="model_version",
        markers=True,
        title=f"Model Precision Proxy ({grain_label})",
    )
    precision_fig.update_yaxes(range=[0, 1])
    precision_fig.update_layout(template="plotly_dark", height=360)
    st.plotly_chart(precision_fig, use_container_width=True)

    st.dataframe(model_df, use_container_width=True)


def _render_rule_effectiveness(rule_df: pd.DataFrame) -> None:
    """Render rule effectiveness table and chart."""
    st.markdown("### Rule Effectiveness Metrics")
    if rule_df.empty:
        st.info("No rule effectiveness data available.")
        return

    fig = px.scatter(
        rule_df,
        x="triggered_count",
        y="precision_proxy",
        size="confirmed_fraud",
        color="rule_category",
        hover_name="rule_name",
        title="Rule Precision vs Trigger Volume",
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(rule_df, use_container_width=True)

    st.download_button(
        "Export Rule Effectiveness (CSV)",
        data=dataframe_to_csv_bytes(rule_df),
        file_name=f"rule_effectiveness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def render(engine: Engine) -> None:
    """Render trend analysis dashboard page."""
    st.markdown("## Trend Analysis")
    st.caption("Analyze fraud trend direction, category concentration, and model/rule performance.")

    filters = _render_filters()
    trends = _fetch_fraud_trends(engine, filters)
    merchant_df, channel_df, geography_df = _fetch_category_breakdown(engine, filters)
    model_df = _fetch_model_performance(engine, filters)
    rule_df = _fetch_rule_effectiveness(engine, filters)

    _render_fraud_trends(trends, filters["grain_label"])
    st.markdown("---")
    _render_category_charts(merchant_df, channel_df, geography_df)
    st.markdown("---")
    _render_model_performance(model_df, filters["grain_label"])
    st.markdown("---")
    _render_rule_effectiveness(rule_df)
