"""Investigation Console page for fraud alert triage and resolution workflows."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text
from sqlalchemy.engine import Engine

from dashboards.streamlit.components.tables import (
    dataframe_to_csv_bytes,
    paginate_dataframe,
    render_alert_queue_table,
    render_pagination_controls,
)

logger = logging.getLogger(__name__)

ALERT_SEVERITIES = ["all", "low", "medium", "high", "critical"]
ALERT_STATUSES = ["all", "open", "investigating", "resolved", "false_positive"]
ALERT_TYPES = ["all", "rule_based", "anomaly", "ml_score", "ensemble"]
SEARCH_FIELDS = {
    "All": "all",
    "Transaction ID": "transaction",
    "Account ID": "account",
    "Customer ID": "customer",
}


def _run_query(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Execute SQL and return rows as a DataFrame."""
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        rows = result.fetchall()
        columns = list(result.keys())
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def _build_alert_where_clause(filters: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build filterable WHERE clause for alert queue and export queries."""
    clauses = ["fa.created_at BETWEEN :start_ts AND :end_ts"]
    params: dict[str, Any] = {
        "start_ts": filters["start_ts"],
        "end_ts": filters["end_ts"],
    }

    if filters.get("severity"):
        clauses.append("fa.severity = :severity")
        params["severity"] = filters["severity"]

    if filters.get("status"):
        clauses.append("fa.status = :status")
        params["status"] = filters["status"]

    if filters.get("alert_type"):
        clauses.append("fa.alert_type = :alert_type")
        params["alert_type"] = filters["alert_type"]

    search_text = filters.get("search_text")
    search_field = filters.get("search_field")
    if search_text:
        params["search_like"] = f"%{search_text.strip()}%"
        if search_field == "transaction":
            clauses.append("CAST(t.transaction_id AS TEXT) ILIKE :search_like")
        elif search_field == "account":
            clauses.append("t.account_id ILIKE :search_like")
        elif search_field == "customer":
            clauses.append("t.customer_id ILIKE :search_like")
        else:
            clauses.append(
                "(CAST(t.transaction_id AS TEXT) ILIKE :search_like "
                "OR t.account_id ILIKE :search_like "
                "OR t.customer_id ILIKE :search_like)"
            )

    return " AND ".join(clauses), params


def _fetch_alert_queue_page(
    engine: Engine,
    filters: dict[str, Any],
    *,
    page: int,
    page_size: int,
) -> tuple[pd.DataFrame, int]:
    """Fetch a page of alert queue rows and total row count."""
    where_sql, params = _build_alert_where_clause(filters)

    count_sql = f"""
        SELECT COUNT(*) AS total_rows
        FROM fraud_alerts fa
        JOIN transactions t ON t.transaction_id = fa.transaction_id
        WHERE {where_sql}
    """
    count_df = _run_query(engine, count_sql, params)
    total_rows = int(count_df.iloc[0]["total_rows"]) if not count_df.empty else 0

    offset = max(page - 1, 0) * page_size
    rows_sql = f"""
        SELECT
            fa.alert_id,
            fa.created_at AS alert_created_at,
            fa.severity,
            fa.status,
            fa.alert_type,
            COALESCE(fa.risk_score, rs.overall_score, 0) AS risk_score,
            fa.rule_id,
            t.transaction_id,
            t.account_id,
            t.customer_id,
            t.transaction_amount,
            t.channel,
            t.geo_country,
            t.merchant_name,
            fa.assigned_to
        FROM fraud_alerts fa
        JOIN transactions t ON t.transaction_id = fa.transaction_id
        LEFT JOIN LATERAL (
            SELECT rs2.overall_score
            FROM risk_scores rs2
            WHERE rs2.transaction_id = t.transaction_id
            ORDER BY rs2.scoring_timestamp DESC
            LIMIT 1
        ) rs ON TRUE
        WHERE {where_sql}
        ORDER BY
            CASE fa.severity
                WHEN 'critical' THEN 1
                WHEN 'high' THEN 2
                WHEN 'medium' THEN 3
                ELSE 4
            END,
            fa.created_at DESC
        LIMIT :limit_rows OFFSET :offset_rows
    """
    rows_params = {**params, "limit_rows": page_size, "offset_rows": offset}
    return _run_query(engine, rows_sql, rows_params), total_rows


def _fetch_alert_queue_export(engine: Engine, filters: dict[str, Any], limit_rows: int = 50000) -> pd.DataFrame:
    """Fetch filtered alerts for CSV export, capped for operational safety."""
    where_sql, params = _build_alert_where_clause(filters)
    sql = f"""
        SELECT
            fa.alert_id,
            fa.created_at AS alert_created_at,
            fa.severity,
            fa.status,
            fa.alert_type,
            COALESCE(fa.risk_score, rs.overall_score, 0) AS risk_score,
            fa.rule_id,
            t.transaction_id,
            t.account_id,
            t.customer_id,
            t.transaction_amount,
            t.transaction_currency,
            t.transaction_type,
            t.channel,
            t.geo_country,
            t.geo_city,
            t.merchant_name,
            fa.assigned_to,
            fa.resolved_at,
            fa.resolution_notes
        FROM fraud_alerts fa
        JOIN transactions t ON t.transaction_id = fa.transaction_id
        LEFT JOIN LATERAL (
            SELECT rs2.overall_score
            FROM risk_scores rs2
            WHERE rs2.transaction_id = t.transaction_id
            ORDER BY rs2.scoring_timestamp DESC
            LIMIT 1
        ) rs ON TRUE
        WHERE {where_sql}
        ORDER BY fa.created_at DESC
        LIMIT :limit_rows
    """
    return _run_query(engine, sql, {**params, "limit_rows": limit_rows})


def _fetch_alert_detail(engine: Engine, alert_id: str) -> pd.DataFrame:
    """Fetch enriched alert detail for selected alert."""
    sql = """
        SELECT
            fa.alert_id,
            fa.transaction_id,
            fa.alert_type,
            fa.rule_id,
            fa.risk_score,
            fa.severity,
            fa.status,
            fa.description,
            fa.details,
            fa.assigned_to,
            fa.resolution_notes,
            fa.created_at,
            fa.updated_at,
            fa.resolved_at,
            t.external_transaction_id,
            t.account_id,
            t.customer_id,
            t.merchant_id,
            t.merchant_name,
            t.merchant_category_code,
            t.transaction_amount,
            t.transaction_currency,
            t.transaction_type,
            t.channel,
            t.card_type,
            t.card_last_four,
            t.ip_address,
            t.device_id,
            t.device_type,
            t.geo_latitude,
            t.geo_longitude,
            t.geo_country,
            t.geo_city,
            t.is_international,
            t.transaction_timestamp,
            t.status AS transaction_status
        FROM fraud_alerts fa
        JOIN transactions t ON t.transaction_id = fa.transaction_id
        WHERE fa.alert_id = :alert_id
        LIMIT 1
    """
    return _run_query(engine, sql, {"alert_id": alert_id})


def _fetch_customer_history(engine: Engine, customer_id: str, transaction_ts: datetime) -> pd.DataFrame:
    """Load customer transaction history in the 30-day lookback window."""
    start_ts = transaction_ts - timedelta(days=30)
    sql = """
        SELECT
            transaction_id,
            transaction_timestamp,
            transaction_amount,
            transaction_currency,
            transaction_type,
            channel,
            merchant_name,
            geo_country,
            status
        FROM transactions
        WHERE customer_id = :customer_id
          AND transaction_timestamp BETWEEN :start_ts AND :end_ts
        ORDER BY transaction_timestamp DESC
        LIMIT 500
    """
    return _run_query(
        engine,
        sql,
        {
            "customer_id": customer_id,
            "start_ts": start_ts,
            "end_ts": transaction_ts,
        },
    )


def _fetch_risk_breakdown(engine: Engine, transaction_id: str) -> pd.DataFrame:
    """Get latest risk score breakdown for a transaction."""
    sql = """
        SELECT
            score_id,
            model_version,
            overall_score,
            rule_score,
            anomaly_score,
            ml_score,
            feature_contributions,
            scoring_timestamp,
            latency_ms
        FROM risk_scores
        WHERE transaction_id = :transaction_id
        ORDER BY scoring_timestamp DESC
        LIMIT 1
    """
    return _run_query(engine, sql, {"transaction_id": transaction_id})


def _fetch_similar_alerts(engine: Engine, alert_id: str, transaction_id: str, customer_id: str) -> pd.DataFrame:
    """Find similar historical alerts by customer and transaction attributes."""
    sql = """
        SELECT
            fa.alert_id,
            fa.alert_type,
            fa.severity,
            fa.status,
            fa.rule_id,
            fa.risk_score,
            fa.created_at,
            t.transaction_amount,
            t.channel,
            t.geo_country,
            t.merchant_name
        FROM fraud_alerts fa
        JOIN transactions t ON t.transaction_id = fa.transaction_id
        WHERE fa.alert_id <> :alert_id
          AND (
              t.customer_id = :customer_id
              OR fa.transaction_id = :transaction_id
              OR fa.rule_id = (
                  SELECT fa2.rule_id
                  FROM fraud_alerts fa2
                  WHERE fa2.alert_id = :alert_id
                  LIMIT 1
              )
          )
          AND fa.created_at >= NOW() - INTERVAL '180 days'
        ORDER BY fa.created_at DESC
        LIMIT 50
    """
    return _run_query(
        engine,
        sql,
        {
            "alert_id": alert_id,
            "transaction_id": transaction_id,
            "customer_id": customer_id,
        },
    )


def _fetch_analyst_workload(engine: Engine, analyst: str) -> pd.DataFrame:
    """Compute workload and throughput metrics for the current analyst."""
    sql = """
        SELECT
            COUNT(*) FILTER (WHERE status = 'open') AS open_alerts,
            COUNT(*) FILTER (WHERE status = 'investigating') AS investigating_alerts,
            COUNT(*) FILTER (WHERE status IN ('resolved', 'false_positive')) AS closed_alerts,
            COUNT(*) FILTER (WHERE assigned_to = :analyst AND status = 'investigating') AS my_active,
            COUNT(*) FILTER (
                WHERE assigned_to = :analyst
                  AND status IN ('resolved', 'false_positive')
                  AND resolved_at >= NOW() - INTERVAL '7 days'
            ) AS my_closed_7d,
            COALESCE(
                AVG(EXTRACT(EPOCH FROM (resolved_at - created_at)) / 3600)
                FILTER (WHERE status IN ('resolved', 'false_positive') AND resolved_at IS NOT NULL),
                0
            ) AS avg_resolution_hours
        FROM fraud_alerts
    """
    return _run_query(engine, sql, {"analyst": analyst})


def _append_investigation_note(engine: Engine, alert_id: str, analyst: str, note: str) -> None:
    """Append investigation note JSON to details.investigation_notes."""
    note_json = json.dumps(
        [
            {
                "analyst": analyst,
                "note": note,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
        ]
    )

    update_sql = """
        UPDATE fraud_alerts
        SET
            details = jsonb_set(
                COALESCE(details, '{}'::jsonb),
                '{investigation_notes}',
                COALESCE(details->'investigation_notes', '[]'::jsonb) || CAST(:note_entry AS jsonb),
                true
            ),
            assigned_to = COALESCE(assigned_to, :analyst),
            updated_at = NOW()
        WHERE alert_id = :alert_id
    """
    audit_sql = """
        INSERT INTO audit_logs (event_type, entity_type, entity_id, action, actor, details)
        VALUES (:event_type, :entity_type, :entity_id, :action, :actor, CAST(:details AS jsonb))
    """

    with engine.begin() as conn:
        conn.execute(
            text(update_sql),
            {"note_entry": note_json, "analyst": analyst, "alert_id": alert_id},
        )
        conn.execute(
            text(audit_sql),
            {
                "event_type": "investigation_note_added",
                "entity_type": "fraud_alert",
                "entity_id": alert_id,
                "action": "add_note",
                "actor": analyst,
                "details": json.dumps({"note": note}),
            },
        )


def _update_alert_status(
    engine: Engine,
    *,
    alert_id: str,
    analyst: str,
    new_status: str,
    resolution_notes: str | None = None,
    details_patch: dict[str, Any] | None = None,
) -> None:
    """Transition alert status and persist audit log in a single transaction."""
    details_patch_json = json.dumps(details_patch or {})

    update_sql = """
        UPDATE fraud_alerts
        SET
            status = :new_status,
            assigned_to = :analyst,
            resolution_notes = CASE
                WHEN :resolution_notes IS NULL OR :resolution_notes = '' THEN resolution_notes
                ELSE :resolution_notes
            END,
            resolved_at = CASE
                WHEN :new_status IN ('resolved', 'false_positive') THEN NOW()
                ELSE resolved_at
            END,
            details = COALESCE(details, '{}'::jsonb) || CAST(:details_patch AS jsonb),
            updated_at = NOW()
        WHERE alert_id = :alert_id
    """
    audit_sql = """
        INSERT INTO audit_logs (event_type, entity_type, entity_id, action, actor, details)
        VALUES (:event_type, :entity_type, :entity_id, :action, :actor, CAST(:details AS jsonb))
    """

    with engine.begin() as conn:
        conn.execute(
            text(update_sql),
            {
                "alert_id": alert_id,
                "analyst": analyst,
                "new_status": new_status,
                "resolution_notes": resolution_notes or "",
                "details_patch": details_patch_json,
            },
        )
        conn.execute(
            text(audit_sql),
            {
                "event_type": "fraud_alert_status_updated",
                "entity_type": "fraud_alert",
                "entity_id": alert_id,
                "action": f"set_status_{new_status}",
                "actor": analyst,
                "details": json.dumps(
                    {
                        "new_status": new_status,
                        "resolution_notes": resolution_notes,
                        "details_patch": details_patch or {},
                    }
                ),
            },
        )


def _render_geo_view(detail_row: pd.Series) -> None:
    """Render geographic context for selected transaction."""
    lat = detail_row.get("geo_latitude")
    lon = detail_row.get("geo_longitude")
    country = detail_row.get("geo_country")
    city = detail_row.get("geo_city")

    if pd.notna(lat) and pd.notna(lon):
        fig = go.Figure(
            data=[
                go.Scattergeo(
                    lat=[float(lat)],
                    lon=[float(lon)],
                    text=[f"{city or 'Unknown city'}, {country or 'Unknown country'}"],
                    mode="markers",
                    marker={
                        "size": 14,
                        "color": "#e74c3c",
                        "line": {"color": "white", "width": 1},
                    },
                    hovertemplate="%{text}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Transaction Location",
            geo={
                "showland": True,
                "landcolor": "#1f2a44",
                "showocean": True,
                "oceancolor": "#0d1b2a",
                "countrycolor": "#526d82",
                "projection_type": "natural earth",
            },
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
            template="plotly_dark",
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)
    elif country:
        df = pd.DataFrame({"geo_country": [country], "txn_count": [1]})
        fig = px.choropleth(
            df,
            locations="geo_country",
            color="txn_count",
            locationmode="ISO-3",
            title="Transaction Country",
            color_continuous_scale="Reds",
        )
        fig.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No geographic coordinates available for this transaction.")


def _render_filters() -> dict[str, Any]:
    """Render page-level filters and search controls."""
    st.markdown("### Alert Queue")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        severity_choice = st.selectbox("Severity", ALERT_SEVERITIES, index=0)
    with c2:
        status_choice = st.selectbox("Status", ALERT_STATUSES, index=0)
    with c3:
        type_choice = st.selectbox("Type", ALERT_TYPES, index=0)
    with c4:
        page_size = st.selectbox("Rows per page", [25, 50, 100, 200], index=1)

    date_col1, date_col2, search_col1, search_col2 = st.columns([1, 1, 1, 2])
    with date_col1:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30))
    with date_col2:
        end_date = st.date_input("End Date", value=date.today())
    with search_col1:
        search_label = st.selectbox("Search By", options=list(SEARCH_FIELDS.keys()))
    with search_col2:
        search_text = st.text_input("Search", placeholder="Transaction/account/customer identifier")

    start_ts = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    end_ts = datetime.combine(end_date, time.max, tzinfo=timezone.utc)

    return {
        "severity": None if severity_choice == "all" else severity_choice,
        "status": None if status_choice == "all" else status_choice,
        "alert_type": None if type_choice == "all" else type_choice,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "search_field": SEARCH_FIELDS[search_label],
        "search_text": search_text.strip(),
        "page_size": int(page_size),
    }


def _render_workload_metrics(engine: Engine, analyst: str) -> None:
    """Render analyst and queue health metrics."""
    metrics_df = _fetch_analyst_workload(engine, analyst)
    if metrics_df.empty:
        st.info("No analyst metrics available.")
        return

    row = metrics_df.iloc[0]
    cols = st.columns(6)
    cols[0].metric("Open", int(row["open_alerts"]))
    cols[1].metric("Investigating", int(row["investigating_alerts"]))
    cols[2].metric("Closed", int(row["closed_alerts"]))
    cols[3].metric("My Active", int(row["my_active"]))
    cols[4].metric("My Closed (7d)", int(row["my_closed_7d"]))
    cols[5].metric("Avg Resolve (hrs)", f"{float(row['avg_resolution_hours']):.2f}")


def _render_detail_view(engine: Engine, analyst: str, alert_id: str) -> None:
    """Render full investigation context and action controls."""
    detail_df = _fetch_alert_detail(engine, alert_id)
    if detail_df.empty:
        st.warning("Selected alert no longer exists.")
        return

    row = detail_df.iloc[0]
    st.markdown(f"### Alert Detail: {row['alert_id']}")

    hdr1, hdr2, hdr3, hdr4, hdr5 = st.columns(5)
    hdr1.metric("Severity", str(row["severity"]))
    hdr2.metric("Status", str(row["status"]))
    hdr3.metric("Type", str(row["alert_type"]))
    hdr4.metric("Risk Score", f"{float(row['risk_score'] or 0):.4f}")
    hdr5.metric("Assigned To", str(row.get("assigned_to") or "unassigned"))

    tabs = st.tabs(
        [
            "Transaction Details",
            "Customer History (30d)",
            "Risk Breakdown",
            "Similar Alerts",
            "Geography",
            "Actions",
        ]
    )

    with tabs[0]:
        left, right = st.columns(2)
        with left:
            st.json(
                {
                    "transaction_id": str(row["transaction_id"]),
                    "external_transaction_id": row.get("external_transaction_id"),
                    "account_id": row.get("account_id"),
                    "customer_id": row.get("customer_id"),
                    "transaction_amount": float(row.get("transaction_amount") or 0),
                    "transaction_currency": row.get("transaction_currency"),
                    "transaction_type": row.get("transaction_type"),
                    "channel": row.get("channel"),
                    "transaction_status": row.get("transaction_status"),
                    "transaction_timestamp": str(row.get("transaction_timestamp")),
                }
            )
        with right:
            st.json(
                {
                    "merchant_id": row.get("merchant_id"),
                    "merchant_name": row.get("merchant_name"),
                    "merchant_category_code": row.get("merchant_category_code"),
                    "geo_country": row.get("geo_country"),
                    "geo_city": row.get("geo_city"),
                    "ip_address": row.get("ip_address"),
                    "device_id": row.get("device_id"),
                    "device_type": row.get("device_type"),
                    "is_international": bool(row.get("is_international")),
                    "rule_id": row.get("rule_id"),
                }
            )

    with tabs[1]:
        customer_history = _fetch_customer_history(
            engine,
            customer_id=str(row["customer_id"]),
            transaction_ts=row["transaction_timestamp"],
        )
        if customer_history.empty:
            st.info("No customer history found for the last 30 days.")
        else:
            history_kpi_1, history_kpi_2, history_kpi_3 = st.columns(3)
            history_kpi_1.metric("Transactions", len(customer_history.index))
            history_kpi_2.metric(
                "Total Amount",
                f"${float(customer_history['transaction_amount'].sum()):,.2f}",
            )
            history_kpi_3.metric(
                "Flagged Count",
                int((customer_history["status"] == "flagged").sum()),
            )
            st.dataframe(customer_history, use_container_width=True)

    with tabs[2]:
        risk_df = _fetch_risk_breakdown(engine, transaction_id=str(row["transaction_id"]))
        if risk_df.empty:
            st.info("No risk score breakdown found.")
        else:
            score_row = risk_df.iloc[0]
            contrib_cols = st.columns(4)
            contrib_cols[0].metric("Overall", f"{float(score_row['overall_score']):.4f}")
            contrib_cols[1].metric("Rule", f"{float(score_row['rule_score'] or 0):.4f}")
            contrib_cols[2].metric("Anomaly", f"{float(score_row['anomaly_score'] or 0):.4f}")
            contrib_cols[3].metric("ML", f"{float(score_row['ml_score'] or 0):.4f}")

            breakdown = pd.DataFrame(
                {
                    "component": ["rule", "anomaly", "ml"],
                    "score": [
                        float(score_row["rule_score"] or 0),
                        float(score_row["anomaly_score"] or 0),
                        float(score_row["ml_score"] or 0),
                    ],
                }
            )
            fig = px.bar(
                breakdown,
                x="component",
                y="score",
                title="Risk Score Contributions",
                color="component",
                color_discrete_sequence=["#3498db", "#f39c12", "#e74c3c"],
            )
            fig.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig, use_container_width=True)

            feature_contribs = score_row.get("feature_contributions") or {}
            if isinstance(feature_contribs, dict) and feature_contribs:
                contrib_df = pd.DataFrame(
                    [(k, v) for k, v in feature_contribs.items()],
                    columns=["feature", "contribution"],
                ).sort_values("contribution", key=lambda s: s.abs(), ascending=False)
                st.markdown("Top Feature Contributions")
                st.dataframe(contrib_df.head(20), use_container_width=True)
            else:
                st.caption("No feature contribution payload available.")

    with tabs[3]:
        similar_df = _fetch_similar_alerts(
            engine,
            alert_id=str(row["alert_id"]),
            transaction_id=str(row["transaction_id"]),
            customer_id=str(row["customer_id"]),
        )
        if similar_df.empty:
            st.info("No similar historical alerts found.")
        else:
            st.dataframe(similar_df, use_container_width=True)

    with tabs[4]:
        _render_geo_view(row)

    with tabs[5]:
        action_col_1, action_col_2 = st.columns(2)

        with action_col_1:
            if st.button("Mark as Investigating", key=f"investigating_{alert_id}"):
                try:
                    _update_alert_status(
                        engine,
                        alert_id=alert_id,
                        analyst=analyst,
                        new_status="investigating",
                        details_patch={"workflow_stage": "manual_investigation"},
                    )
                    st.success("Alert moved to investigating.")
                    st.rerun()
                except Exception as exc:
                    logger.exception("Failed to update alert status", exc_info=exc)
                    st.error("Could not update alert status.")

            escalate_target = st.selectbox(
                "Escalate To",
                options=["L2 Fraud Ops", "Risk Lead", "Compliance"],
                key=f"escalate_target_{alert_id}",
            )
            escalate_reason = st.text_area(
                "Escalation Reason",
                placeholder="Describe why this alert needs escalation",
                key=f"escalate_reason_{alert_id}",
            )
            if st.button("Escalate", key=f"escalate_{alert_id}"):
                if not escalate_reason.strip():
                    st.warning("Escalation reason is required.")
                else:
                    try:
                        _update_alert_status(
                            engine,
                            alert_id=alert_id,
                            analyst=analyst,
                            new_status="investigating",
                            details_patch={
                                "escalated": True,
                                "escalation_target": escalate_target,
                                "escalation_reason": escalate_reason.strip(),
                                "escalated_at": datetime.now(tz=timezone.utc).isoformat(),
                            },
                        )
                        st.success("Alert escalated successfully.")
                        st.rerun()
                    except Exception as exc:
                        logger.exception("Failed to escalate alert", exc_info=exc)
                        st.error("Could not escalate alert.")

        with action_col_2:
            outcome = st.radio(
                "Resolution",
                options=["confirmed_fraud", "false_positive"],
                horizontal=True,
                key=f"resolution_outcome_{alert_id}",
            )
            resolution_note = st.text_area(
                "Resolution Notes",
                placeholder="Add investigation findings and evidence",
                key=f"resolution_notes_{alert_id}",
            )
            if st.button("Resolve Alert", key=f"resolve_{alert_id}"):
                if not resolution_note.strip():
                    st.warning("Resolution notes are required.")
                else:
                    try:
                        new_status = "resolved" if outcome == "confirmed_fraud" else "false_positive"
                        _update_alert_status(
                            engine,
                            alert_id=alert_id,
                            analyst=analyst,
                            new_status=new_status,
                            resolution_notes=resolution_note.strip(),
                            details_patch={"resolution_type": outcome},
                        )
                        st.success("Alert resolved successfully.")
                        st.rerun()
                    except Exception as exc:
                        logger.exception("Failed to resolve alert", exc_info=exc)
                        st.error("Could not resolve alert.")

            note_text = st.text_area(
                "Add Investigation Note",
                placeholder="Capture intermediate findings",
                key=f"note_text_{alert_id}",
            )
            if st.button("Add Note", key=f"add_note_{alert_id}"):
                if not note_text.strip():
                    st.warning("Note text is required.")
                else:
                    try:
                        _append_investigation_note(engine, alert_id, analyst, note_text.strip())
                        st.success("Note added successfully.")
                        st.rerun()
                    except Exception as exc:
                        logger.exception("Failed to add investigation note", exc_info=exc)
                        st.error("Could not add note.")


def render(engine: Engine) -> None:
    """Render investigation console page."""
    st.markdown("## Investigation Console")
    st.caption("Review, triage, investigate, and resolve fraud alerts from a single console.")

    analyst = str(st.session_state.get("username") or "analyst")

    _render_workload_metrics(engine, analyst)
    st.markdown("---")

    filters = _render_filters()

    default_page = int(st.session_state.get("investigation_page", 1))
    queue_df, total_rows = _fetch_alert_queue_page(
        engine,
        filters,
        page=default_page,
        page_size=filters["page_size"],
    )

    st.caption(f"Filtered alerts: {total_rows:,}")

    # Calculate pages from total count, then allow user to navigate.
    pager_source = pd.DataFrame(index=range(total_rows))
    pager = paginate_dataframe(pager_source, page_number=default_page, page_size=filters["page_size"])
    selected_page = render_pagination_controls(
        key_prefix="investigation_queue",
        total_pages=pager.total_pages,
        default_page=default_page,
    )

    if selected_page != default_page:
        st.session_state["investigation_page"] = selected_page
        st.rerun()

    st.session_state["investigation_page"] = selected_page

    render_alert_queue_table(queue_df)

    export_df = _fetch_alert_queue_export(engine, filters)
    st.download_button(
        "Export Filtered Alerts (CSV)",
        data=dataframe_to_csv_bytes(export_df),
        file_name=f"riskpulse_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if queue_df.empty:
        return

    selected_alert = st.selectbox(
        "Select alert for investigation",
        options=queue_df["alert_id"].astype(str).tolist(),
        index=0,
    )
    _render_detail_view(engine, analyst, selected_alert)
