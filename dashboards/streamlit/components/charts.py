"""Reusable Plotly chart components for RiskPulse dashboards.

Provides standardized, interactive chart builders with consistent
color coding by risk level and responsive layouts.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RISK_COLOR_MAP: dict[str, str] = {
    "low": "#2ecc71",
    "medium": "#f39c12",
    "high": "#e74c3c",
    "critical": "#8e44ad",
}

SEVERITY_ORDER = ["low", "medium", "high", "critical"]

STATUS_COLOR_MAP: dict[str, str] = {
    "approved": "#2ecc71",
    "pending": "#f39c12",
    "flagged": "#e74c3c",
    "declined": "#95a5a6",
}

CHART_TEMPLATE = "plotly_dark"

_LAYOUT_DEFAULTS: dict[str, Any] = {
    "template": CHART_TEMPLATE,
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#ecf0f1", "size": 12},
    "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
}


def _apply_defaults(fig: go.Figure, title: str | None = None, height: int = 350) -> go.Figure:
    """Apply common layout defaults to a Plotly figure."""
    updates: dict[str, Any] = {**_LAYOUT_DEFAULTS, "height": height}
    if title:
        updates["title"] = {"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 16}}
    fig.update_layout(**updates)
    return fig


def transaction_volume_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart of transaction volume over time.

    Expects columns: ``time_bucket``, ``txn_count``.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["time_bucket"],
            y=df["txn_count"],
            mode="lines+markers",
            line={"color": "#3498db", "width": 2},
            marker={"size": 4},
            fill="tozeroy",
            fillcolor="rgba(52,152,219,0.15)",
            hovertemplate="<b>%{x}</b><br>Transactions: %{y}<extra></extra>",
        )
    )
    fig = _apply_defaults(fig, title="Transaction Volume Over Time")
    fig.update_xaxes(title_text="Time", showgrid=False)
    fig.update_yaxes(title_text="Count", showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    return fig


def fraud_rate_gauge(fraud_rate: float) -> go.Figure:
    """Gauge chart showing current fraud rate percentage."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=fraud_rate,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 10], "ticksuffix": "%"},
                "bar": {"color": "#e74c3c"},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 1], "color": "rgba(46,204,113,0.3)"},
                    {"range": [1, 3], "color": "rgba(243,156,18,0.3)"},
                    {"range": [3, 10], "color": "rgba(231,76,60,0.3)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 2},
                    "thickness": 0.8,
                    "value": fraud_rate,
                },
            },
            title={"text": "Real-Time Fraud Rate", "font": {"size": 16}},
        )
    )
    fig = _apply_defaults(fig, height=280)
    return fig


def risk_score_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of risk score distribution.

    Expects column: ``overall_score``.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df["overall_score"],
            nbinsx=20,
            marker={
                "color": df["overall_score"] if len(df) > 0 else [],
                "colorscale": [
                    [0.0, "#2ecc71"],
                    [0.3, "#f39c12"],
                    [0.7, "#e74c3c"],
                    [1.0, "#8e44ad"],
                ],
                "line": {"width": 0.5, "color": "rgba(255,255,255,0.3)"},
            },
            hovertemplate="Score Range: %{x}<br>Count: %{y}<extra></extra>",
        )
    )
    fig = _apply_defaults(fig, title="Risk Score Distribution")
    fig.update_xaxes(title_text="Risk Score", range=[0, 1], showgrid=False)
    fig.update_yaxes(title_text="Frequency", showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    return fig


def alert_severity_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart of alert severity breakdown.

    Expects columns: ``severity``, ``count``.
    """
    colors = [RISK_COLOR_MAP.get(sev, "#95a5a6") for sev in df["severity"]]
    fig = go.Figure(
        go.Pie(
            labels=df["severity"],
            values=df["count"],
            marker={"colors": colors, "line": {"color": "#1a1a2e", "width": 2}},
            textinfo="label+percent",
            textfont={"size": 12},
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            hole=0.45,
        )
    )
    fig = _apply_defaults(fig, title="Alert Severity Breakdown", height=320)
    return fig


def geo_heatmap(df: pd.DataFrame) -> go.Figure:
    """Geographic heatmap of transaction locations.

    Expects columns: ``geo_country``, ``txn_count``.
    """
    fig = go.Figure(
        go.Choropleth(
            locations=df["geo_country"],
            z=df["txn_count"],
            colorscale="YlOrRd",
            locationmode="ISO-3",
            marker_line_color="rgba(255,255,255,0.2)",
            marker_line_width=0.5,
            colorbar={
                "title": "Transactions",
                "tickfont": {"color": "#ecf0f1"},
                "titlefont": {"color": "#ecf0f1"},
            },
            hovertemplate="<b>%{location}</b><br>Transactions: %{z}<extra></extra>",
        )
    )
    fig = _apply_defaults(fig, title="Transaction Heatmap by Country", height=400)
    fig.update_geos(
        bgcolor="rgba(0,0,0,0)",
        landcolor="rgba(30,30,60,0.6)",
        oceancolor="rgba(10,10,40,0.4)",
        showocean=True,
        showlakes=False,
        showcountries=True,
        countrycolor="rgba(255,255,255,0.15)",
    )
    return fig


def channel_breakdown_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart of transaction counts by channel.

    Expects columns: ``channel``, ``txn_count``, ``fraud_count``.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=df["channel"],
            y=df["txn_count"],
            name="Total",
            marker_color="#3498db",
            hovertemplate="%{x}: %{y} transactions<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=df["channel"],
            y=df["fraud_count"],
            name="Fraudulent",
            marker_color="#e74c3c",
            hovertemplate="%{x}: %{y} flagged<extra></extra>",
        ),
        secondary_y=False,
    )
    fig = _apply_defaults(fig, title="Transactions by Channel")
    fig.update_layout(barmode="group", legend={"orientation": "h", "y": -0.15})
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    return fig


def kpi_card_html(
    title: str,
    value: str,
    delta: str | None = None,
    delta_positive: bool | None = None,
    icon: str = "📊",
) -> str:
    """Generate HTML for a KPI metric card.

    Returns an HTML string suitable for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta_positive else "▼"
        color = "#2ecc71" if delta_positive else "#e74c3c"
        delta_html = f'<div class="kpi-delta" style="color:{color}">{arrow} {delta}</div>'

    return f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-content">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
    </div>
    """


def live_feed_table_html(df: pd.DataFrame) -> str:
    """Render a styled HTML table for the live transaction feed.

    Expects columns: ``transaction_id``, ``transaction_amount``,
    ``status``, ``risk_score``, ``channel``, ``geo_country``,
    ``transaction_timestamp``.
    """
    if df.empty:
        return '<div class="feed-empty">No recent transactions</div>'

    rows: list[str] = []
    for _, row in df.iterrows():
        score = float(row.get("risk_score", 0) or 0)
        if score >= 0.85:
            risk_class = "risk-critical"
        elif score >= 0.6:
            risk_class = "risk-high"
        elif score >= 0.3:
            risk_class = "risk-medium"
        else:
            risk_class = "risk-low"

        status = str(row.get("status", ""))
        status_color = STATUS_COLOR_MAP.get(status, "#95a5a6")
        txn_id = str(row.get("transaction_id", ""))[:8]
        amount = f"${float(row.get('transaction_amount', 0)):,.2f}"
        ts = str(row.get("transaction_timestamp", ""))[:19]

        rows.append(
            f"<tr>"
            f'<td class="feed-id">{txn_id}…</td>'
            f"<td>{amount}</td>"
            f'<td><span class="status-badge" style="background:{status_color}">{status}</span></td>'
            f'<td><span class="risk-badge {risk_class}">{score:.3f}</span></td>'
            f"<td>{row.get('channel', '')}</td>"
            f"<td>{row.get('geo_country', '')}</td>"
            f"<td>{ts}</td>"
            f"</tr>"
        )

    return f"""
    <div class="feed-table-wrapper">
        <table class="feed-table">
            <thead>
                <tr>
                    <th>ID</th><th>Amount</th><th>Status</th>
                    <th>Risk</th><th>Channel</th><th>Country</th><th>Time</th>
                </tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
    """
