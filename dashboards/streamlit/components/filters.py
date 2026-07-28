"""Reusable filter components for RiskPulse Streamlit dashboards.

Provides sidebar filter widgets that return structured filter dictionaries
consumed by the data-access layer.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import streamlit as st

SEVERITY_OPTIONS = ["all", "low", "medium", "high", "critical"]
STATUS_OPTIONS = ["all", "pending", "approved", "declined", "flagged"]
CHANNEL_OPTIONS = ["all", "online", "pos", "atm", "mobile"]
ALERT_STATUS_OPTIONS = ["all", "open", "investigating", "resolved", "false_positive"]
TRANSACTION_TYPE_OPTIONS = ["all", "purchase", "withdrawal", "transfer", "refund"]

TIME_RANGE_PRESETS: dict[str, timedelta] = {
    "Last 15 minutes": timedelta(minutes=15),
    "Last 1 hour": timedelta(hours=1),
    "Last 6 hours": timedelta(hours=6),
    "Last 24 hours": timedelta(hours=24),
    "Last 7 days": timedelta(days=7),
    "Last 30 days": timedelta(days=30),
}


def render_time_range_filter(key_prefix: str = "monitor") -> tuple[datetime, datetime]:
    """Render a time-range selector and return ``(start, end)`` UTC datetimes."""
    preset = st.selectbox(
        "Time Range",
        options=list(TIME_RANGE_PRESETS.keys()),
        index=1,
        key=f"{key_prefix}_time_range",
    )
    now = datetime.now(tz=timezone.utc)
    delta = TIME_RANGE_PRESETS[preset]
    return now - delta, now


def render_severity_filter(key_prefix: str = "monitor") -> str | None:
    """Render severity filter; returns ``None`` for 'all'."""
    choice = st.selectbox(
        "Severity",
        options=SEVERITY_OPTIONS,
        index=0,
        key=f"{key_prefix}_severity",
    )
    return None if choice == "all" else choice


def render_status_filter(key_prefix: str = "monitor") -> str | None:
    """Render transaction status filter; returns ``None`` for 'all'."""
    choice = st.selectbox(
        "Status",
        options=STATUS_OPTIONS,
        index=0,
        key=f"{key_prefix}_status",
    )
    return None if choice == "all" else choice


def render_channel_filter(key_prefix: str = "monitor") -> str | None:
    """Render channel filter; returns ``None`` for 'all'."""
    choice = st.selectbox(
        "Channel",
        options=CHANNEL_OPTIONS,
        index=0,
        key=f"{key_prefix}_channel",
    )
    return None if choice == "all" else choice


def render_risk_score_slider(key_prefix: str = "monitor") -> tuple[float, float]:
    """Render a risk-score range slider; returns ``(min, max)``."""
    return st.slider(
        "Risk Score Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05,
        key=f"{key_prefix}_risk_range",
    )


def render_sidebar_filters(key_prefix: str = "monitor") -> dict[str, Any]:
    """Render a complete set of sidebar filters and return a filter dict.

    Keys returned:
    - ``time_start``, ``time_end`` — UTC datetimes
    - ``severity`` — str | None
    - ``status`` — str | None
    - ``channel`` — str | None
    - ``risk_min``, ``risk_max`` — floats 0-1
    """
    st.sidebar.markdown("### 🔍 Filters")

    with st.sidebar:
        time_start, time_end = render_time_range_filter(key_prefix)
        severity = render_severity_filter(key_prefix)
        status = render_status_filter(key_prefix)
        channel = render_channel_filter(key_prefix)
        risk_min, risk_max = render_risk_score_slider(key_prefix)

    return {
        "time_start": time_start,
        "time_end": time_end,
        "severity": severity,
        "status": status,
        "channel": channel,
        "risk_min": risk_min,
        "risk_max": risk_max,
    }
