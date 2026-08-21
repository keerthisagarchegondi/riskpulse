"""Model evaluation framework for anomaly detection models.

Provides comprehensive evaluation metrics, threshold analysis,
and performance reporting for fraud detection models.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for a model."""

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    true_positive_rate: float = 0.0
    accuracy: float = 0.0
    specificity: float = 0.0
    confusion: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "true_positive_rate": self.true_positive_rate,
            "accuracy": self.accuracy,
            "specificity": self.specificity,
            "confusion_matrix": self.confusion,
        }

    def passes_thresholds(
        self,
        min_precision: float = 0.80,
        min_recall: float = 0.85,
        max_fpr: float = 0.03,
    ) -> bool:
        """Check if metrics meet production thresholds."""
        return (
            self.precision >= min_precision
            and self.recall >= min_recall
            and self.false_positive_rate <= max_fpr
        )


@dataclass
class ThresholdAnalysis:
    """Analysis of model performance across different decision thresholds."""

    thresholds: list[float] = field(default_factory=list)
    precisions: list[float] = field(default_factory=list)
    recalls: list[float] = field(default_factory=list)
    f1_scores: list[float] = field(default_factory=list)
    fprs: list[float] = field(default_factory=list)
    optimal_threshold: float = 0.0
    optimal_f1: float = 0.0


@dataclass
class LatencyMetrics:
    """Prediction latency statistics."""

    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    min_ms: float = 0.0
    samples: int = 0

    def meets_sla(self, max_latency_ms: float = 10.0) -> bool:
        return self.p99_ms <= max_latency_ms


class ModelEvaluator:
    """Evaluates anomaly detection model performance."""

    def __init__(self, model: Any) -> None:
        """Initialize evaluator with a trained model.

        Args:
            model: Trained AnomalyDetector instance.
        """
        self._model = model

    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
    ) -> EvaluationMetrics:
        """Compute full evaluation metrics on a labeled dataset.

        Args:
            X: Feature DataFrame.
            y_true: Binary labels (1=fraud, 0=legitimate).

        Returns:
            EvaluationMetrics with all computed values.
        """
        results = self._model.predict_batch(X)
        y_pred = np.array([1 if r.is_anomaly else 0 for r in results])
        scores = np.array([-r.anomaly_score for r in results])

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        try:
            auc_roc = roc_auc_score(y_true, scores)
        except ValueError:
            auc_roc = 0.0

        try:
            auc_pr = average_precision_score(y_true, scores)
        except ValueError:
            auc_pr = 0.0

        total = tn + fp + fn + tp
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)
        tpr = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        accuracy = (tp + tn) / max(total, 1)

        metrics = EvaluationMetrics(
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            auc_roc=float(auc_roc),
            auc_pr=float(auc_pr),
            false_positive_rate=float(fpr),
            false_negative_rate=float(fnr),
            true_positive_rate=float(tpr),
            accuracy=float(accuracy),
            specificity=float(specificity),
            confusion={
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        )

        logger.info(
            "Evaluation: precision=%.4f, recall=%.4f, F1=%.4f, AUC=%.4f, FPR=%.4f",
            metrics.precision,
            metrics.recall,
            metrics.f1_score,
            metrics.auc_roc,
            metrics.false_positive_rate,
        )
        return metrics

    def threshold_analysis(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        n_thresholds: int = 50,
    ) -> ThresholdAnalysis:
        """Analyze model performance across different score thresholds.

        Args:
            X: Feature DataFrame.
            y_true: Binary labels.
            n_thresholds: Number of threshold values to evaluate.

        Returns:
            ThresholdAnalysis with per-threshold metrics.
        """
        scores = self._model.get_anomaly_scores(X)
        # Lower scores = more anomalous for Isolation Forest
        # Negate for standard threshold analysis
        neg_scores = -scores

        thresholds = np.linspace(
            np.percentile(neg_scores, 1),
            np.percentile(neg_scores, 99),
            n_thresholds,
        )

        analysis = ThresholdAnalysis()
        best_f1 = 0.0

        for thresh in thresholds:
            preds = (neg_scores >= thresh).astype(int)

            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            fpr_val = (preds[y_true == 0].sum()) / max((y_true == 0).sum(), 1)

            analysis.thresholds.append(float(thresh))
            analysis.precisions.append(float(prec))
            analysis.recalls.append(float(rec))
            analysis.f1_scores.append(float(f1))
            analysis.fprs.append(float(fpr_val))

            if f1 > best_f1:
                best_f1 = f1
                analysis.optimal_threshold = float(thresh)
                analysis.optimal_f1 = float(f1)

        logger.info(
            "Threshold analysis: optimal=%.4f (F1=%.4f)",
            analysis.optimal_threshold,
            analysis.optimal_f1,
        )
        return analysis

    def benchmark_latency(
        self,
        X: pd.DataFrame,
        n_iterations: int = 100,
    ) -> LatencyMetrics:
        """Benchmark single-prediction latency.

        Args:
            X: DataFrame to sample transactions from.
            n_iterations: Number of prediction iterations.

        Returns:
            LatencyMetrics with timing statistics.
        """
        sample_row = X.iloc[0].to_dict()
        latencies = []

        # Warmup
        for _ in range(10):
            self._model.predict(sample_row)

        for _ in range(n_iterations):
            start = time.perf_counter()
            self._model.predict(sample_row)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        latencies_arr = np.array(latencies)

        metrics = LatencyMetrics(
            mean_ms=float(np.mean(latencies_arr)),
            median_ms=float(np.median(latencies_arr)),
            p95_ms=float(np.percentile(latencies_arr, 95)),
            p99_ms=float(np.percentile(latencies_arr, 99)),
            max_ms=float(np.max(latencies_arr)),
            min_ms=float(np.min(latencies_arr)),
            samples=n_iterations,
        )

        logger.info(
            "Latency: mean=%.3f ms, p99=%.3f ms, meets_sla=%s",
            metrics.mean_ms,
            metrics.p99_ms,
            metrics.meets_sla(),
        )
        return metrics

    def generate_report(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive evaluation report.

        Args:
            X: Feature DataFrame.
            y_true: Binary labels.
            output_path: Optional path to save JSON report.

        Returns:
            Full evaluation report as dict.
        """
        logger.info("Generating comprehensive evaluation report...")

        metrics = self.evaluate(X, y_true)
        threshold_results = self.threshold_analysis(X, y_true)
        latency = self.benchmark_latency(X)

        report = {
            "model_version": self._model.model_version,
            "evaluation_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "dataset": {
                "total_samples": len(X),
                "fraud_samples": int(y_true.sum()),
                "legitimate_samples": int((y_true == 0).sum()),
                "fraud_ratio": float(y_true.mean()),
            },
            "metrics": metrics.to_dict(),
            "threshold_analysis": {
                "optimal_threshold": threshold_results.optimal_threshold,
                "optimal_f1": threshold_results.optimal_f1,
            },
            "latency": {
                "mean_ms": latency.mean_ms,
                "median_ms": latency.median_ms,
                "p95_ms": latency.p95_ms,
                "p99_ms": latency.p99_ms,
                "meets_sla_10ms": latency.meets_sla(10.0),
            },
            "production_readiness": {
                "precision_threshold_met": metrics.precision >= 0.80,
                "recall_threshold_met": metrics.recall >= 0.85,
                "fpr_threshold_met": metrics.false_positive_rate <= 0.03,
                "latency_sla_met": latency.meets_sla(10.0),
                "overall_ready": (metrics.passes_thresholds() and latency.meets_sla(10.0)),
            },
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info("Report saved to %s", output_path)

        return report
