"""Model performance monitoring page for RiskPulse."""

from __future__ import annotations

import json
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

DEFAULT_THRESHOLD = 0.7
PSI_BINS = np.linspace(0.0, 1.0, 11)


def auc(x: pd.Series | np.ndarray | list[float], y: pd.Series | np.ndarray | list[float]) -> float:
    """Calculate trapezoidal area under a monotonic curve."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 2:
        return 0.0
    order = np.argsort(x_arr)
    sorted_x = x_arr[order]
    sorted_y = y_arr[order]
    return float(np.sum(np.diff(sorted_x) * (sorted_y[:-1] + sorted_y[1:]) / 2.0))


def confusion_matrix(
    actual: pd.Series | np.ndarray | list[int],
    predicted: pd.Series | np.ndarray | list[int],
    *,
    labels: list[int],
) -> np.ndarray:
    """Build a confusion matrix for known labels."""
    actual_arr = np.asarray(actual, dtype=int)
    predicted_arr = np.asarray(predicted, dtype=int)
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_index = {label: idx for idx, label in enumerate(labels)}
    for actual_value, predicted_value in zip(actual_arr, predicted_arr):
        if actual_value in label_index and predicted_value in label_index:
            matrix[label_index[actual_value], label_index[predicted_value]] += 1
    return matrix


def roc_curve(
    actual: pd.Series | np.ndarray | list[int],
    scores: pd.Series | np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate ROC curve points for binary labels."""
    actual_arr = np.asarray(actual, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    thresholds = np.r_[np.inf, np.sort(np.unique(score_arr))[::-1]]
    positives = max(int((actual_arr == 1).sum()), 1)
    negatives = max(int((actual_arr == 0).sum()), 1)
    tpr: list[float] = []
    fpr: list[float] = []
    for threshold in thresholds:
        predicted = score_arr >= threshold
        tp = int(((actual_arr == 1) & predicted).sum())
        fp = int(((actual_arr == 0) & predicted).sum())
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
    return np.asarray(fpr), np.asarray(tpr), thresholds


def precision_recall_curve(
    actual: pd.Series | np.ndarray | list[int],
    scores: pd.Series | np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate precision-recall curve points for binary labels."""
    actual_arr = np.asarray(actual, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    thresholds = np.sort(np.unique(score_arr))[::-1]
    positives = max(int((actual_arr == 1).sum()), 1)
    precision: list[float] = [1.0]
    recall: list[float] = [0.0]
    for threshold in thresholds:
        predicted = score_arr >= threshold
        tp = int(((actual_arr == 1) & predicted).sum())
        fp = int(((actual_arr == 0) & predicted).sum())
        precision.append(tp / (tp + fp) if tp + fp else 1.0)
        recall.append(tp / positives)
    return np.asarray(precision), np.asarray(recall), thresholds


def roc_auc_score(
    actual: pd.Series | np.ndarray | list[int],
    scores: pd.Series | np.ndarray | list[float],
) -> float:
    """Calculate ROC AUC for binary labels."""
    fpr, tpr, _ = roc_curve(actual, scores)
    return auc(fpr, tpr)


def _run_query(engine: Engine, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Execute SQL and return records as a DataFrame."""
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        rows = result.fetchall()
        cols = list(result.keys())
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _render_filters() -> dict[str, Any]:
    """Render page-level model monitoring filters."""
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        start_date = st.date_input(
            "Start Date",
            value=date.today() - timedelta(days=90),
            key="model_perf_start",
        )
    with c2:
        end_date = st.date_input("End Date", value=date.today(), key="model_perf_end")
    with c3:
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.05,
            max_value=0.95,
            value=DEFAULT_THRESHOLD,
            step=0.05,
        )
    with c4:
        grain = st.selectbox("Trend Grain", options=["day", "week", "month"], index=0)

    return {
        "start_ts": datetime.combine(start_date, time.min, tzinfo=timezone.utc),
        "end_ts": datetime.combine(end_date, time.max, tzinfo=timezone.utc),
        "threshold": float(threshold),
        "grain": grain,
    }


def _fetch_scored_labels(engine: Engine, filters: dict[str, Any]) -> pd.DataFrame:
    """Fetch scores with analyst-confirmed labels where labels are available."""
    sql = """
        SELECT
            rs.transaction_id,
            rs.model_version,
            rs.overall_score::FLOAT AS overall_score,
            rs.ml_score::FLOAT AS ml_score,
            rs.latency_ms,
            rs.scoring_timestamp,
            t.status AS transaction_status,
            fa.status AS alert_status,
            CASE
                WHEN fa.status = 'resolved' THEN 1
                WHEN fa.status = 'false_positive' THEN 0
                WHEN fa.alert_id IS NULL AND t.status IN ('approved', 'declined') THEN 0
                ELSE NULL
            END AS actual_label
        FROM risk_scores rs
        JOIN transactions t ON t.transaction_id = rs.transaction_id
        LEFT JOIN fraud_alerts fa ON fa.transaction_id = rs.transaction_id
        WHERE rs.scoring_timestamp BETWEEN :start_ts AND :end_ts
        ORDER BY rs.scoring_timestamp
        LIMIT 50000
    """
    df = _run_query(engine, sql, filters)
    if not df.empty:
        df["overall_score"] = pd.to_numeric(df["overall_score"], errors="coerce").fillna(0.0)
        df["actual_label"] = pd.to_numeric(df["actual_label"], errors="coerce")
        df["scoring_timestamp"] = pd.to_datetime(df["scoring_timestamp"], utc=True)
    return df


def _fetch_feature_importance(engine: Engine, filters: dict[str, Any]) -> pd.DataFrame:
    """Aggregate absolute feature contribution magnitude from score payloads."""
    sql = """
        SELECT
            feature.key AS feature,
            AVG(ABS(feature.value::FLOAT)) AS mean_abs_contribution,
            COUNT(*) AS observations
        FROM risk_scores rs
        CROSS JOIN LATERAL jsonb_each_text(COALESCE(rs.feature_contributions, '{}'::jsonb)) feature
        WHERE rs.scoring_timestamp BETWEEN :start_ts AND :end_ts
          AND feature.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
        GROUP BY feature.key
        HAVING COUNT(*) >= 3
        ORDER BY mean_abs_contribution DESC
        LIMIT 25
    """
    return _run_query(engine, sql, filters)


def _fetch_model_registry(engine: Engine) -> pd.DataFrame:
    """Fetch model registry rows for comparison and release context."""
    sql = """
        SELECT
            model_name,
            model_version,
            model_type,
            status,
            metrics,
            trained_at,
            deployed_at
        FROM model_registry
        ORDER BY COALESCE(deployed_at, trained_at, created_at) DESC
    """
    return _run_query(engine, sql)


def _fetch_score_windows(engine: Engine, filters: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    """Fetch current and previous score windows for drift analysis."""
    window = filters["end_ts"] - filters["start_ts"]
    params = {
        "current_start": filters["start_ts"],
        "current_end": filters["end_ts"],
        "baseline_start": filters["start_ts"] - window,
        "baseline_end": filters["start_ts"],
    }
    sql = """
        SELECT 'current' AS window_name, overall_score::FLOAT AS score
        FROM risk_scores
        WHERE scoring_timestamp BETWEEN :current_start AND :current_end
        UNION ALL
        SELECT 'baseline' AS window_name, overall_score::FLOAT AS score
        FROM risk_scores
        WHERE scoring_timestamp BETWEEN :baseline_start AND :baseline_end
    """
    df = _run_query(engine, sql, params)
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return (
        pd.to_numeric(df.loc[df["window_name"] == "baseline", "score"], errors="coerce").dropna(),
        pd.to_numeric(df.loc[df["window_name"] == "current", "score"], errors="coerce").dropna(),
    )


def _labeled_frame(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return rows with known labels and predicted labels."""
    if df.empty:
        return pd.DataFrame(columns=[*df.columns, "predicted_label"])
    labeled = df.dropna(subset=["actual_label", "overall_score"]).copy()
    labeled["actual_label"] = labeled["actual_label"].astype(int)
    labeled["predicted_label"] = (labeled["overall_score"] >= threshold).astype(int)
    return labeled


def build_confusion_matrix(df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Build a 2x2 confusion matrix from labeled model scores."""
    labeled = _labeled_frame(df, threshold)
    labels = [0, 1]
    matrix = confusion_matrix(
        labeled["actual_label"] if not labeled.empty else [],
        labeled["predicted_label"] if not labeled.empty else [],
        labels=labels,
    )
    return pd.DataFrame(
        matrix,
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"],
    )


def build_auc_trend(
    df: pd.DataFrame,
    *,
    grain: str = "day",
    min_labels: int = 5,
) -> pd.DataFrame:
    """Calculate ROC AUC over time by model version."""
    labeled = _labeled_frame(df, DEFAULT_THRESHOLD)
    if labeled.empty:
        return pd.DataFrame(columns=["period", "model_version", "auc", "labeled_count"])

    grouped = labeled.groupby(
        [pd.Grouper(key="scoring_timestamp", freq=_grain_to_frequency(grain)), "model_version"]
    )
    rows: list[dict[str, Any]] = []
    for (period, model_version), group in grouped:
        if len(group.index) < min_labels or group["actual_label"].nunique() < 2:
            continue
        rows.append(
            {
                "period": period,
                "model_version": model_version,
                "auc": float(roc_auc_score(group["actual_label"], group["overall_score"])),
                "labeled_count": int(len(group.index)),
            }
        )
    return pd.DataFrame(rows)


def build_precision_recall_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average precision by model version."""
    labeled = _labeled_frame(df, DEFAULT_THRESHOLD)
    rows: list[dict[str, Any]] = []
    for model_version, group in labeled.groupby("model_version"):
        if len(group.index) < 2 or group["actual_label"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(group["actual_label"], group["overall_score"])
        rows.append(
            {
                "model_version": model_version,
                "average_precision": float(auc(recall, precision)),
                "labeled_count": int(len(group.index)),
            }
        )
    return pd.DataFrame(rows)


def calculate_population_stability_index(
    baseline: pd.Series | np.ndarray | list[float],
    current: pd.Series | np.ndarray | list[float],
    bins: np.ndarray = PSI_BINS,
) -> float:
    """Calculate population stability index for score drift monitoring."""
    baseline_arr = np.asarray(pd.Series(baseline).dropna(), dtype=float)
    current_arr = np.asarray(pd.Series(current).dropna(), dtype=float)
    if len(baseline_arr) == 0 or len(current_arr) == 0:
        return 0.0

    base_counts, _ = np.histogram(baseline_arr, bins=bins)
    current_counts, _ = np.histogram(current_arr, bins=bins)
    epsilon = 1e-8
    base_pct = (base_counts + epsilon) / (base_counts.sum() + epsilon * len(base_counts))
    current_pct = (current_counts + epsilon) / (
        current_counts.sum() + epsilon * len(current_counts)
    )
    return float(np.sum((current_pct - base_pct) * np.log(current_pct / base_pct)))


def build_degradation_alerts(
    labeled_scores: pd.DataFrame,
    *,
    baseline_scores: pd.Series | None = None,
    current_scores: pd.Series | None = None,
    auc_floor: float = 0.75,
    latency_p95_ms: float = 250.0,
    psi_warning: float = 0.10,
) -> pd.DataFrame:
    """Create model monitoring alerts from observed performance metrics."""
    alerts: list[dict[str, Any]] = []
    labeled = _labeled_frame(labeled_scores, DEFAULT_THRESHOLD)

    if not labeled.empty and labeled["actual_label"].nunique() == 2:
        current_auc = float(roc_auc_score(labeled["actual_label"], labeled["overall_score"]))
        if current_auc < auc_floor:
            alerts.append(
                {
                    "severity": "critical" if current_auc < auc_floor - 0.10 else "warning",
                    "metric": "roc_auc",
                    "value": current_auc,
                    "threshold": auc_floor,
                    "message": "ROC AUC is below the production floor.",
                }
            )

    if "latency_ms" in labeled_scores.columns:
        latency = pd.to_numeric(labeled_scores["latency_ms"], errors="coerce").dropna()
        if not latency.empty:
            p95 = float(np.percentile(latency, 95))
            if p95 > latency_p95_ms:
                alerts.append(
                    {
                        "severity": "warning",
                        "metric": "p95_latency_ms",
                        "value": p95,
                        "threshold": latency_p95_ms,
                        "message": "Model scoring latency is above target.",
                    }
                )

    if baseline_scores is not None and current_scores is not None:
        psi = calculate_population_stability_index(baseline_scores, current_scores)
        if psi > psi_warning:
            alerts.append(
                {
                    "severity": "critical" if psi > 0.25 else "warning",
                    "metric": "score_psi",
                    "value": psi,
                    "threshold": psi_warning,
                    "message": "Prediction score distribution drift detected.",
                }
            )

    return pd.DataFrame(alerts, columns=["severity", "metric", "value", "threshold", "message"])


def _grain_to_frequency(grain: str) -> str:
    return {"day": "D", "week": "W", "month": "MS"}.get(grain, "D")


def _registry_metrics_table(registry_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize registry JSON metrics into tabular columns."""
    if registry_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in registry_df.iterrows():
        metrics = row.get("metrics") or {}
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except json.JSONDecodeError:
                metrics = {}
        rows.append(
            {
                "model_name": row.get("model_name"),
                "model_version": row.get("model_version"),
                "model_type": row.get("model_type"),
                "status": row.get("status"),
                "auc": _metric_value(metrics, "auc", "roc_auc"),
                "precision": _metric_value(metrics, "precision"),
                "recall": _metric_value(metrics, "recall"),
                "f1": _metric_value(metrics, "f1", "f1_score"),
                "deployed_at": row.get("deployed_at"),
            }
        )
    return pd.DataFrame(rows)


def _metric_value(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                return None
    return None


def _render_confusion_matrix(df: pd.DataFrame, threshold: float) -> None:
    matrix = build_confusion_matrix(df, threshold)
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        aspect="auto",
    )
    fig.update_layout(template="plotly_dark", height=360)
    st.plotly_chart(fig, use_container_width=True)


def _render_roc_pr_curves(df: pd.DataFrame) -> None:
    labeled = _labeled_frame(df, DEFAULT_THRESHOLD)
    roc_fig = go.Figure()
    pr_fig = go.Figure()

    for model_version, group in labeled.groupby("model_version"):
        if len(group.index) < 2 or group["actual_label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(group["actual_label"], group["overall_score"])
        precision, recall, _ = precision_recall_curve(group["actual_label"], group["overall_score"])
        roc_auc = roc_auc_score(group["actual_label"], group["overall_score"])
        pr_auc = auc(recall, precision)
        roc_fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model_version} AUC {roc_auc:.3f}")
        )
        pr_fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{model_version} AP {pr_auc:.3f}",
            )
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
    for fig, title, x_title, y_title in [
        (roc_fig, "ROC Curves", "False Positive Rate", "True Positive Rate"),
        (pr_fig, "Precision-Recall Curves", "Recall", "Precision"),
    ]:
        fig.update_layout(template="plotly_dark", title=title, height=380)
        fig.update_xaxes(title=x_title, range=[0, 1])
        fig.update_yaxes(title=y_title, range=[0, 1])

    c1, c2 = st.columns(2)
    c1.plotly_chart(roc_fig, use_container_width=True)
    c2.plotly_chart(pr_fig, use_container_width=True)


def _render_score_distribution(df: pd.DataFrame) -> None:
    labeled = _labeled_frame(df, DEFAULT_THRESHOLD)
    if labeled.empty:
        st.info("No labeled score distribution available.")
        return

    labeled["label"] = np.where(labeled["actual_label"] == 1, "Fraud", "Legitimate")
    fig = px.histogram(
        labeled,
        x="overall_score",
        color="label",
        barmode="overlay",
        nbins=30,
        opacity=0.75,
        title="Score Distribution by Actual Label",
        color_discrete_map={"Fraud": "#e74c3c", "Legitimate": "#3498db"},
    )
    fig.update_layout(template="plotly_dark", height=360)
    fig.update_xaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def _render_feature_importance(feature_df: pd.DataFrame) -> None:
    if feature_df.empty:
        st.info("No feature contribution payloads available for this window.")
        return

    chart_df = feature_df.sort_values("mean_abs_contribution", ascending=True)
    fig = px.bar(
        chart_df,
        x="mean_abs_contribution",
        y="feature",
        orientation="h",
        title="Feature Importance from Mean Absolute Contribution",
        color="mean_abs_contribution",
        color_continuous_scale="Teal",
    )
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)


def _render_auc_trend(df: pd.DataFrame, grain: str) -> None:
    trend = build_auc_trend(df, grain=grain)
    if trend.empty:
        st.info("Not enough labeled observations to calculate historical AUC.")
        return
    fig = px.line(
        trend,
        x="period",
        y="auc",
        color="model_version",
        markers=True,
        title="ROC AUC Tracking Over Time",
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(template="plotly_dark", height=360)
    st.plotly_chart(fig, use_container_width=True)


def _render_model_comparison(labeled_df: pd.DataFrame, registry_df: pd.DataFrame) -> None:
    registry_metrics = _registry_metrics_table(registry_df)
    pr_summary = build_precision_recall_summary(labeled_df)

    observed_rows: list[dict[str, Any]] = []
    labeled = _labeled_frame(labeled_df, DEFAULT_THRESHOLD)
    for model_version, group in labeled.groupby("model_version"):
        if group["actual_label"].nunique() < 2:
            continue
        matrix = build_confusion_matrix(group, DEFAULT_THRESHOLD)
        tp = int(matrix.loc["Actual Fraud", "Predicted Fraud"])
        fp = int(matrix.loc["Actual Legitimate", "Predicted Fraud"])
        fn = int(matrix.loc["Actual Fraud", "Predicted Legitimate"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        observed_rows.append(
            {
                "model_version": model_version,
                "observed_auc": float(roc_auc_score(group["actual_label"], group["overall_score"])),
                "observed_precision": precision,
                "observed_recall": recall,
                "labeled_count": int(len(group.index)),
            }
        )
    observed = pd.DataFrame(observed_rows)
    if not observed.empty and not pr_summary.empty:
        observed = observed.merge(pr_summary, on=["model_version", "labeled_count"], how="left")

    if registry_metrics.empty and observed.empty:
        st.info("No registry or observed model comparison data available.")
        return

    comparison = (
        registry_metrics.merge(observed, on="model_version", how="outer")
        if not registry_metrics.empty and not observed.empty
        else registry_metrics if not registry_metrics.empty else observed
    )
    st.dataframe(comparison, use_container_width=True)

    metric_cols = [col for col in ["auc", "observed_auc", "average_precision"] if col in comparison]
    if metric_cols:
        plot_df = comparison.melt(
            id_vars=["model_version"],
            value_vars=metric_cols,
            var_name="metric",
            value_name="value",
        ).dropna()
        fig = px.bar(
            plot_df,
            x="model_version",
            y="value",
            color="metric",
            barmode="group",
            title="Model A/B Comparison",
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig, use_container_width=True)


def _render_drift(baseline_scores: pd.Series, current_scores: pd.Series) -> float:
    psi = calculate_population_stability_index(baseline_scores, current_scores)
    hist_df = pd.DataFrame(
        {
            "score": pd.concat([baseline_scores, current_scores], ignore_index=True),
            "window": ["Baseline"] * len(baseline_scores) + ["Current"] * len(current_scores),
        }
    )
    if hist_df.empty:
        st.info("No score windows available for drift visualization.")
        return psi

    fig = px.histogram(
        hist_df,
        x="score",
        color="window",
        barmode="overlay",
        nbins=20,
        opacity=0.65,
        title=f"Prediction Drift Detection (PSI {psi:.4f})",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(template="plotly_dark", height=360)
    st.plotly_chart(fig, use_container_width=True)
    return psi


def _render_degradation_alerts(alerts: pd.DataFrame) -> None:
    st.markdown("### Performance Degradation Alerts")
    if alerts.empty:
        st.success("No model degradation alerts for the selected window.")
        return
    st.dataframe(alerts, use_container_width=True)


def render(engine: Engine) -> None:
    """Render the model performance monitoring dashboard."""
    st.markdown("## Model Performance Monitoring")
    st.caption("Track classifier quality, score behavior, drift, and production health.")

    filters = _render_filters()
    with st.spinner("Loading model metrics..."):
        scored_df = _fetch_scored_labels(engine, filters)
        feature_df = _fetch_feature_importance(engine, filters)
        registry_df = _fetch_model_registry(engine)
        baseline_scores, current_scores = _fetch_score_windows(engine, filters)

    labeled = _labeled_frame(scored_df, filters["threshold"])
    current_auc = None
    if not labeled.empty and labeled["actual_label"].nunique() == 2:
        current_auc = roc_auc_score(labeled["actual_label"], labeled["overall_score"])
    pr_summary = build_precision_recall_summary(scored_df)
    psi = calculate_population_stability_index(baseline_scores, current_scores)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Scored Transactions", f"{len(scored_df.index):,}")
    k2.metric("Labeled Outcomes", f"{len(labeled.index):,}")
    k3.metric("Current ROC AUC", "N/A" if current_auc is None else f"{current_auc:.3f}")
    k4.metric("Score PSI", f"{psi:.4f}")

    tabs = st.tabs(
        [
            "Performance",
            "Feature Importance",
            "A/B Comparison",
            "Drift",
            "Alerts",
        ]
    )

    with tabs[0]:
        c1, c2 = st.columns([1, 1])
        with c1:
            _render_confusion_matrix(scored_df, filters["threshold"])
        with c2:
            _render_score_distribution(scored_df)
        _render_roc_pr_curves(scored_df)
        _render_auc_trend(scored_df, filters["grain"])
        if not pr_summary.empty:
            st.dataframe(pr_summary, use_container_width=True)

    with tabs[1]:
        _render_feature_importance(feature_df)

    with tabs[2]:
        _render_model_comparison(scored_df, registry_df)

    with tabs[3]:
        _render_drift(baseline_scores, current_scores)

    with tabs[4]:
        alerts = build_degradation_alerts(
            scored_df,
            baseline_scores=baseline_scores,
            current_scores=current_scores,
        )
        _render_degradation_alerts(alerts)
        if not alerts.empty:
            st.download_button(
                "Export Alerts (CSV)",
                data=dataframe_to_csv_bytes(alerts),
                file_name=f"model_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    st.caption(f"Last refreshed: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
