"""Reusable table helpers for Streamlit dashboard pages.

Provides pagination, lightweight formatting, and CSV export utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class TablePage:
    """Represents the current page view for a paginated DataFrame."""

    rows: pd.DataFrame
    page_number: int
    page_size: int
    total_rows: int
    total_pages: int


_SEVERITY_COLOR = {
    "low": "#2ecc71",
    "medium": "#f39c12",
    "high": "#e74c3c",
    "critical": "#8e44ad",
}

_STATUS_COLOR = {
    "open": "#e67e22",
    "investigating": "#3498db",
    "resolved": "#2ecc71",
    "false_positive": "#95a5a6",
}


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to UTF-8 CSV bytes for download controls."""
    return df.to_csv(index=False).encode("utf-8")


def paginate_dataframe(
    df: pd.DataFrame,
    *,
    page_number: int,
    page_size: int,
) -> TablePage:
    """Slice a DataFrame into a stable page payload."""
    total_rows = len(df.index)
    if total_rows == 0:
        return TablePage(
            rows=df,
            page_number=1,
            page_size=page_size,
            total_rows=0,
            total_pages=1,
        )

    total_pages = max((total_rows + page_size - 1) // page_size, 1)
    safe_page = min(max(page_number, 1), total_pages)
    start = (safe_page - 1) * page_size
    end = start + page_size

    return TablePage(
        rows=df.iloc[start:end].copy(),
        page_number=safe_page,
        page_size=page_size,
        total_rows=total_rows,
        total_pages=total_pages,
    )


def render_pagination_controls(
    *,
    key_prefix: str,
    total_pages: int,
    default_page: int = 1,
) -> int:
    """Render compact pagination controls and return selected page number."""
    if total_pages <= 1:
        st.caption("Page 1 of 1")
        return 1

    cols = st.columns([1, 2, 1])
    with cols[0]:
        prev_clicked = st.button("Previous", key=f"{key_prefix}_prev")
    with cols[1]:
        selected = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=min(max(default_page, 1), total_pages),
            step=1,
            key=f"{key_prefix}_page_select",
        )
        st.caption(f"Page {selected} of {total_pages}")
    with cols[2]:
        next_clicked = st.button("Next", key=f"{key_prefix}_next")

    if prev_clicked:
        return max(int(selected) - 1, 1)
    if next_clicked:
        return min(int(selected) + 1, total_pages)
    return int(selected)


def render_alert_queue_table(df: pd.DataFrame) -> None:
    """Render alert queue table with severity/status highlighting."""
    if df.empty:
        st.info("No alerts matched the current filters.")
        return

    styled = (
        df.style.format(
            {
                "transaction_amount": "${:,.2f}",
                "risk_score": "{:.4f}",
            }
        )
        .map(
            lambda v: (
                f"background-color: {_SEVERITY_COLOR.get(str(v), '#2c3e50')}; color: white;"
                if pd.notna(v)
                else ""
            ),
            subset=["severity"],
        )
        .map(
            lambda v: (
                f"background-color: {_STATUS_COLOR.get(str(v), '#34495e')}; color: white;"
                if pd.notna(v)
                else ""
            ),
            subset=["status"],
        )
    )
    st.dataframe(styled, use_container_width=True)
