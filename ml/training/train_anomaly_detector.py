"""Training pipeline for Isolation Forest anomaly detection model.

Handles data preparation, time-based splitting, hyperparameter tuning,
cross-validation, and model persistence with full reproducibility.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detection.anomaly_detector import ANOMALY_FEATURES, AnomalyDetector  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "ml" / "models" / "isolation_forest"
DEFAULT_RANDOM_STATE = 42

HYPERPARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_samples": [256, 512, "auto"],
    "contamination": [0.01, 0.02, 0.03, 0.05],
    "max_features": [0.5, 0.8, 1.0],
}


def generate_synthetic_data(
    n_samples: int = 10000,
    fraud_ratio: float = 0.02,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic transaction data with realistic fraud patterns.

    Args:
        n_samples: Total number of transactions.
        fraud_ratio: Fraction of fraudulent transactions.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (feature DataFrame, binary labels array).
    """
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit_data = {
        "transaction_amount": rng.lognormal(mean=3.5, sigma=1.0, size=n_legit),
        "transaction_count_1hour": rng.poisson(lam=2, size=n_legit),
        "transaction_count_24hour": rng.poisson(lam=8, size=n_legit),
        "amount_mean_24hour": rng.lognormal(mean=3.5, sigma=0.5, size=n_legit),
        "amount_std_24hour": rng.exponential(scale=20, size=n_legit),
        "time_since_last_transaction_seconds": rng.exponential(scale=3600, size=n_legit),
        "distance_from_last_location_km": rng.exponential(scale=10, size=n_legit),
        "unique_merchants_24hour": rng.poisson(lam=3, size=n_legit),
        "unique_countries_24hour": np.ones(n_legit),
        "hour_of_day": rng.integers(6, 23, size=n_legit).astype(float),
        "is_international": rng.binomial(1, 0.05, size=n_legit).astype(float),
        "amount_to_avg_ratio": rng.lognormal(mean=0, sigma=0.3, size=n_legit),
    }

    # Fraudulent transactions - distinct patterns
    fraud_data = {
        "transaction_amount": rng.lognormal(mean=6.0, sigma=1.5, size=n_fraud),
        "transaction_count_1hour": rng.poisson(lam=8, size=n_fraud),
        "transaction_count_24hour": rng.poisson(lam=25, size=n_fraud),
        "amount_mean_24hour": rng.lognormal(mean=5.0, sigma=1.0, size=n_fraud),
        "amount_std_24hour": rng.exponential(scale=100, size=n_fraud),
        "time_since_last_transaction_seconds": rng.exponential(scale=120, size=n_fraud),
        "distance_from_last_location_km": rng.exponential(scale=500, size=n_fraud),
        "unique_merchants_24hour": rng.poisson(lam=10, size=n_fraud),
        "unique_countries_24hour": rng.poisson(lam=3, size=n_fraud).astype(float) + 1,
        "hour_of_day": rng.integers(0, 6, size=n_fraud).astype(float),
        "is_international": rng.binomial(1, 0.7, size=n_fraud).astype(float),
        "amount_to_avg_ratio": rng.lognormal(mean=1.5, sigma=0.8, size=n_fraud),
    }

    legit_df = pd.DataFrame(legit_data)
    fraud_df = pd.DataFrame(fraud_data)

    X = pd.concat([legit_df, fraud_df], ignore_index=True)
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])
    shuffle_idx = rng.permutation(n_samples)
    X = X.iloc[shuffle_idx].reset_index(drop=True)
    y = y[shuffle_idx]

    # Add timestamps for time-based splitting
    base_time = pd.Timestamp("2026-01-01")
    timestamps = pd.date_range(start=base_time, periods=n_samples, freq="5min")
    X["timestamp"] = timestamps
    X["transaction_id"] = [f"TXN-SYNTH-{i:06d}" for i in range(n_samples)]

    return X, y


def time_based_split(
    X: pd.DataFrame,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Split data chronologically to prevent data leakage.

    Args:
        X: Feature DataFrame (must have 'timestamp' column or be sorted by time).
        y: Label array.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X.iloc[:train_end].reset_index(drop=True)
    X_val = X.iloc[train_end:val_end].reset_index(drop=True)
    X_test = X.iloc[val_end:].reset_index(drop=True)

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    logger.info(
        "Time-based split: train=%d (fraud=%.2f%%), val=%d (fraud=%.2f%%), test=%d (fraud=%.2f%%)",
        len(X_train),
        y_train.mean() * 100,
        len(X_val),
        y_val.mean() * 100,
        len(X_test),
        y_test.mean() * 100,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def cross_validate_timeseries(
    X: pd.DataFrame,
    y: np.ndarray,
    n_estimators: int = 200,
    max_samples: int | str = "auto",
    contamination: float = 0.02,
    max_features: float = 0.8,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, list[float]]:
    """Perform time-series cross-validation.

    Uses expanding window splits (no future data leakage).

    Returns:
        Dict of metric name -> list of scores per fold.
    """
    feature_cols = [f for f in ANOMALY_FEATURES if f in X.columns]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    metrics: dict[str, list[float]] = {
        "precision": [],
        "recall": [],
        "f1": [],
        "auc_roc": [],
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold = X.iloc[train_idx][feature_cols]
        X_val_fold = X.iloc[val_idx][feature_cols]
        y_val_fold = y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train_scaled)

        preds = model.predict(X_val_scaled)
        pred_labels = (preds == -1).astype(int)
        scores = -model.decision_function(X_val_scaled)

        prec = precision_score(y_val_fold, pred_labels, zero_division=0)
        rec = recall_score(y_val_fold, pred_labels, zero_division=0)
        f1 = f1_score(y_val_fold, pred_labels, zero_division=0)

        try:
            auc = roc_auc_score(y_val_fold, scores)
        except ValueError:
            auc = 0.0

        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)
        metrics["auc_roc"].append(auc)

        logger.info(
            "Fold %d: precision=%.3f, recall=%.3f, F1=%.3f, AUC=%.3f",
            fold + 1,
            prec,
            rec,
            f1,
            auc,
        )

    return metrics


def hyperparameter_search(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    param_grid: dict[str, list] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Grid search for best hyperparameters using validation set.

    Args:
        X_train: Training features.
        X_val: Validation features.
        y_val: Validation labels.
        param_grid: Parameter grid to search.
        random_state: Random seed.

    Returns:
        Dict with best parameters and their scores.
    """
    if param_grid is None:
        param_grid = HYPERPARAM_GRID

    feature_cols = [f for f in ANOMALY_FEATURES if f in X_train.columns]
    X_train_feat = X_train[feature_cols]
    X_val_feat = X_val[feature_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_val_scaled = scaler.transform(X_val_feat)

    best_f1 = 0.0
    best_params: dict[str, Any] = {}
    results: list[dict[str, Any]] = []

    for n_est in param_grid["n_estimators"]:
        for max_samp in param_grid["max_samples"]:
            for contam in param_grid["contamination"]:
                for max_feat in param_grid["max_features"]:
                    model = IsolationForest(
                        n_estimators=n_est,
                        max_samples=max_samp,
                        contamination=contam,
                        max_features=max_feat,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                    model.fit(X_train_scaled)
                    preds = model.predict(X_val_scaled)
                    pred_labels = (preds == -1).astype(int)

                    prec = precision_score(y_val, pred_labels, zero_division=0)
                    rec = recall_score(y_val, pred_labels, zero_division=0)
                    f1 = f1_score(y_val, pred_labels, zero_division=0)

                    params = {
                        "n_estimators": n_est,
                        "max_samples": max_samp,
                        "contamination": contam,
                        "max_features": max_feat,
                    }
                    results.append({**params, "precision": prec, "recall": rec, "f1": f1})

                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = params

    logger.info("Best params: %s (F1=%.4f)", best_params, best_f1)
    return {"best_params": best_params, "best_f1": best_f1, "all_results": results}


def train_model(
    output_dir: str | Path | None = None,
    n_samples: int = 10000,
    fraud_ratio: float = 0.02,
    random_state: int = DEFAULT_RANDOM_STATE,
    run_hyperparam_search: bool = False,
    n_cv_splits: int = 5,
) -> dict[str, Any]:
    """Full training pipeline: data generation, training, evaluation, and saving.

    Args:
        output_dir: Directory to save model artifacts.
        n_samples: Number of synthetic training samples.
        fraud_ratio: Fraction of fraud in synthetic data.
        random_state: Random seed for reproducibility.
        run_hyperparam_search: Whether to run full hyperparameter search.
        n_cv_splits: Number of time-series CV folds.

    Returns:
        Dict with training results and evaluation metrics.
    """
    if output_dir is None:
        output_dir = DEFAULT_MODEL_OUTPUT

    output_dir = Path(output_dir)
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("ANOMALY DETECTOR TRAINING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Generate synthetic data
    logger.info(
        "Step 1: Generating synthetic data (n=%d, fraud_ratio=%.2f)", n_samples, fraud_ratio
    )
    X, y = generate_synthetic_data(n_samples, fraud_ratio, random_state)
    logger.info("Data shape: %s, Fraud count: %d", X.shape, y.sum())

    # Step 2: Time-based split
    logger.info("Step 2: Time-based train/val/test split")
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(X, y)

    # Step 3: Hyperparameter tuning (optional)
    best_params = {
        "n_estimators": 200,
        "max_samples": "auto",
        "contamination": 0.02,
        "max_features": 0.8,
    }

    if run_hyperparam_search:
        logger.info("Step 3: Hyperparameter search")
        search_results = hyperparameter_search(X_train, X_val, y_val, random_state=random_state)
        best_params = search_results["best_params"]
    else:
        logger.info("Step 3: Using default hyperparameters (skip search)")

    # Step 4: Cross-validation
    logger.info("Step 4: Time-series cross-validation (%d folds)", n_cv_splits)
    cv_metrics = cross_validate_timeseries(
        X_train,
        y_train,
        n_estimators=best_params["n_estimators"],
        max_samples=best_params["max_samples"],
        contamination=best_params["contamination"],
        max_features=best_params["max_features"],
        n_splits=n_cv_splits,
        random_state=random_state,
    )

    cv_summary = {
        metric: {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }
        for metric, scores in cv_metrics.items()
    }
    logger.info("CV Results: %s", cv_summary)

    # Step 5: Train final model on full training set
    logger.info("Step 5: Training final model")
    detector = AnomalyDetector(
        n_estimators=best_params["n_estimators"],
        max_samples=best_params["max_samples"],
        contamination=best_params["contamination"],
        max_features=best_params["max_features"],
        random_state=random_state,
    )
    detector.fit(X_train)

    # Step 6: Evaluate on test set
    logger.info("Step 6: Evaluating on holdout test set")
    results = detector.predict_batch(X_test)
    pred_labels = np.array([1 if r.is_anomaly else 0 for r in results])
    pred_scores = np.array([-r.anomaly_score for r in results])

    test_precision = precision_score(y_test, pred_labels, zero_division=0)
    test_recall = recall_score(y_test, pred_labels, zero_division=0)
    test_f1 = f1_score(y_test, pred_labels, zero_division=0)

    try:
        test_auc = roc_auc_score(y_test, pred_scores)
    except ValueError:
        test_auc = 0.0

    fpr = (pred_labels[y_test == 0].sum()) / max((y_test == 0).sum(), 1)

    logger.info("Test Precision: %.4f", test_precision)
    logger.info("Test Recall: %.4f", test_recall)
    logger.info("Test F1: %.4f", test_f1)
    logger.info("Test AUC-ROC: %.4f", test_auc)
    logger.info("False Positive Rate: %.4f", fpr)

    # Step 7: Latency benchmark
    logger.info("Step 7: Latency benchmark")
    sample_features = X_test.iloc[0].to_dict()
    latencies = []
    for _ in range(100):
        result = detector.predict(sample_features)
        latencies.append(result.prediction_latency_ms)

    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    logger.info("Avg latency: %.3f ms, P99: %.3f ms", avg_latency, p99_latency)

    # Step 8: Save model
    logger.info("Step 8: Saving model to %s", output_dir)
    detector.save(output_dir)

    elapsed = time.time() - start_time
    logger.info("Training complete in %.1f seconds", elapsed)

    evaluation_report = {
        "test_metrics": {
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "auc_roc": test_auc,
            "false_positive_rate": fpr,
        },
        "cv_metrics": cv_summary,
        "best_params": best_params,
        "latency": {
            "avg_ms": float(avg_latency),
            "p99_ms": float(p99_latency),
        },
        "data": {
            "total_samples": n_samples,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "fraud_ratio": fraud_ratio,
        },
        "model_version": detector.model_version,
        "random_state": random_state,
        "training_time_seconds": elapsed,
    }

    # Save evaluation report
    report_path = output_dir / "evaluation_report.json"
    import json

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(evaluation_report, f, indent=2)

    logger.info("Evaluation report saved to %s", report_path)
    return evaluation_report


def main() -> None:
    """CLI entrypoint for model training."""
    parser = argparse.ArgumentParser(description="Train Isolation Forest anomaly detector")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_MODEL_OUTPUT),
        help="Output directory for model artifacts",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of synthetic training samples",
    )
    parser.add_argument(
        "--fraud-ratio",
        type=float,
        default=0.02,
        help="Fraction of fraudulent transactions",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--hyperparam-search",
        action="store_true",
        help="Run full hyperparameter grid search",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of time-series CV folds",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    report = train_model(
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        fraud_ratio=args.fraud_ratio,
        random_state=args.random_state,
        run_hyperparam_search=args.hyperparam_search,
        n_cv_splits=args.cv_splits,
    )

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model Version: {report['model_version']}")
    print(f"Test Precision: {report['test_metrics']['precision']:.4f}")
    print(f"Test Recall: {report['test_metrics']['recall']:.4f}")
    print(f"Test F1: {report['test_metrics']['f1']:.4f}")
    print(f"Test AUC-ROC: {report['test_metrics']['auc_roc']:.4f}")
    print(f"False Positive Rate: {report['test_metrics']['false_positive_rate']:.4f}")
    print(f"Avg Latency: {report['latency']['avg_ms']:.3f} ms")
    print(f"P99 Latency: {report['latency']['p99_ms']:.3f} ms")
    print(f"Training Time: {report['training_time_seconds']:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
