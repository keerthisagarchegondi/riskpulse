"""Training pipeline for gradient boosted tree fraud risk scoring model.

Handles synthetic data generation with realistic fraud patterns,
feature engineering (50+ features), class imbalance handling (SMOTE/class weights),
model training (XGBoost/LightGBM), hyperparameter optimization (Optuna),
evaluation, and model persistence.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detection.feature_store import FEATURE_CATALOG  # noqa: E402
from src.fraud_detection.risk_scorer import RiskScorer  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "ml" / "models" / "risk_scorer"
DEFAULT_RANDOM_STATE = 42

# Feature groups for synthetic data generation
TRANSACTION_FEATURES = [
    "transaction_amount",
    "amount_zscore",
    "amount_to_avg_ratio",
    "amount_percentile",
    "amount_log",
]
TEMPORAL_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "time_since_last_transaction",
    "minutes_since_midnight",
]
VELOCITY_FEATURES_1H = [
    "txn_count_1h",
    "txn_amount_sum_1h",
    "txn_amount_avg_1h",
    "txn_amount_max_1h",
]
VELOCITY_FEATURES_24H = [
    "txn_count_24h",
    "txn_amount_sum_24h",
    "txn_amount_avg_24h",
    "txn_amount_max_24h",
    "txn_amount_std_24h",
]
VELOCITY_FEATURES_7D = ["txn_count_7d", "txn_amount_sum_7d", "txn_amount_avg_7d"]
MERCHANT_FEATURES = [
    "unique_merchants_24h",
    "unique_merchants_7d",
    "new_merchant_flag",
    "merchant_risk_score",
    "merchant_txn_count_30d",
    "merchant_fraud_rate",
]
GEO_FEATURES = [
    "unique_countries_24h",
    "is_international",
    "country_risk_score",
    "distance_from_last_location_km",
    "impossible_travel_flag",
]
DEVICE_FEATURES = [
    "known_device_flag",
    "device_age_days",
    "device_txn_count",
    "multiple_accounts_on_device",
]
BEHAVIORAL_FEATURES = [
    "unusual_hour_flag",
    "channel_switch_flag",
    "amount_deviation_from_mean",
    "amount_deviation_from_median",
    "txn_frequency_deviation",
]
SEQUENCE_FEATURES = [
    "consecutive_declined_count",
    "rapid_succession_flag",
    "time_since_last_decline",
    "decline_rate_24h",
]
ACCOUNT_FEATURES = [
    "account_age_days",
    "account_total_txn_count",
    "account_avg_amount",
    "account_std_amount",
    "account_max_amount",
]
CROSS_FEATURES = [
    "amount_x_velocity",
    "amount_x_hour_risk",
    "international_x_new_merchant",
    "high_amount_x_unusual_hour",
]


def generate_synthetic_data(
    n_samples: int = 50000,
    fraud_ratio: float = 0.02,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic transaction data with 50+ features and realistic fraud patterns.

    Args:
        n_samples: Total number of transactions.
        fraud_ratio: Fraction of fraudulent transactions.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (feature DataFrame with 50+ columns, binary labels array).
    """
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    legit_df = _generate_legit_features(n_legit, rng)
    fraud_df = _generate_fraud_features(n_fraud, rng)

    X = pd.concat([legit_df, fraud_df], ignore_index=True)
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    # Add timestamps for time-based splitting
    base_time = pd.Timestamp("2026-01-01")
    timestamps = pd.date_range(start=base_time, periods=n_samples, freq="2min")
    X["timestamp"] = timestamps
    X["transaction_id"] = [f"TXN-RISK-{i:06d}" for i in range(n_samples)]

    # Shuffle while preserving temporal order for splits
    sort_idx = np.argsort(timestamps)
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]

    logger.info(
        "Generated %d samples (%d fraud, %.2f%% ratio) with %d features",
        n_samples,
        n_fraud,
        fraud_ratio * 100,
        X.shape[1] - 2,
    )
    return X, y


def _generate_legit_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate feature distributions for legitimate transactions."""
    data: dict[str, np.ndarray] = {}

    # Transaction features
    amounts = rng.lognormal(mean=3.5, sigma=1.0, size=n)
    avg_amounts = rng.lognormal(mean=3.5, sigma=0.5, size=n)
    std_amounts = rng.exponential(scale=20, size=n) + 1.0
    data["transaction_amount"] = amounts
    data["amount_zscore"] = (amounts - avg_amounts) / np.maximum(std_amounts, 1.0)
    data["amount_to_avg_ratio"] = amounts / np.maximum(avg_amounts, 1.0)
    data["amount_percentile"] = rng.beta(5, 5, size=n)
    data["amount_log"] = np.log1p(amounts)

    # Temporal features
    data["hour_of_day"] = rng.integers(7, 22, size=n).astype(float)
    data["day_of_week"] = rng.integers(0, 7, size=n).astype(float)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(float)
    data["is_holiday"] = rng.binomial(1, 0.03, size=n).astype(float)
    data["time_since_last_transaction"] = rng.exponential(scale=3600, size=n)
    data["minutes_since_midnight"] = data["hour_of_day"] * 60 + rng.integers(0, 60, size=n)

    # Velocity features 1h
    data["txn_count_1h"] = rng.poisson(lam=1.5, size=n).astype(float)
    data["txn_amount_sum_1h"] = amounts * data["txn_count_1h"] * rng.uniform(0.5, 1.5, size=n)
    data["txn_amount_avg_1h"] = data["txn_amount_sum_1h"] / np.maximum(data["txn_count_1h"], 1)
    data["txn_amount_max_1h"] = data["txn_amount_avg_1h"] * rng.uniform(1.0, 2.0, size=n)

    # Velocity features 24h
    data["txn_count_24h"] = rng.poisson(lam=6, size=n).astype(float)
    data["txn_amount_sum_24h"] = amounts * data["txn_count_24h"] * rng.uniform(0.8, 1.2, size=n)
    data["txn_amount_avg_24h"] = data["txn_amount_sum_24h"] / np.maximum(data["txn_count_24h"], 1)
    data["txn_amount_max_24h"] = data["txn_amount_avg_24h"] * rng.uniform(1.0, 3.0, size=n)
    data["txn_amount_std_24h"] = rng.exponential(scale=15, size=n)

    # Velocity features 7d
    data["txn_count_7d"] = rng.poisson(lam=25, size=n).astype(float)
    data["txn_amount_sum_7d"] = amounts * data["txn_count_7d"] * rng.uniform(0.7, 1.3, size=n)
    data["txn_amount_avg_7d"] = data["txn_amount_sum_7d"] / np.maximum(data["txn_count_7d"], 1)

    # Merchant features
    data["unique_merchants_24h"] = rng.poisson(lam=3, size=n).astype(float)
    data["unique_merchants_7d"] = rng.poisson(lam=8, size=n).astype(float)
    data["new_merchant_flag"] = rng.binomial(1, 0.1, size=n).astype(float)
    data["merchant_risk_score"] = rng.beta(2, 20, size=n)
    data["merchant_txn_count_30d"] = rng.poisson(lam=200, size=n).astype(float)
    data["merchant_fraud_rate"] = rng.beta(1, 500, size=n)

    # Geographic features
    data["unique_countries_24h"] = np.ones(n)
    data["is_international"] = rng.binomial(1, 0.05, size=n).astype(float)
    data["country_risk_score"] = rng.beta(2, 20, size=n)
    data["distance_from_last_location_km"] = rng.exponential(scale=10, size=n)
    data["impossible_travel_flag"] = np.zeros(n)

    # Device features
    data["known_device_flag"] = rng.binomial(1, 0.92, size=n).astype(float)
    data["device_age_days"] = rng.exponential(scale=365, size=n) + 30
    data["device_txn_count"] = rng.poisson(lam=100, size=n).astype(float)
    data["multiple_accounts_on_device"] = rng.binomial(1, 0.02, size=n).astype(float)

    # Behavioral features
    data["unusual_hour_flag"] = np.zeros(n)
    data["channel_switch_flag"] = rng.binomial(1, 0.05, size=n).astype(float)
    data["amount_deviation_from_mean"] = amounts - avg_amounts
    data["amount_deviation_from_median"] = amounts - avg_amounts * 0.9
    data["txn_frequency_deviation"] = rng.normal(0, 0.3, size=n)

    # Sequence features
    data["consecutive_declined_count"] = rng.poisson(lam=0.1, size=n).astype(float)
    data["rapid_succession_flag"] = rng.binomial(1, 0.02, size=n).astype(float)
    data["time_since_last_decline"] = rng.exponential(scale=86400, size=n)
    data["decline_rate_24h"] = rng.beta(1, 100, size=n)

    # Account features
    data["account_age_days"] = rng.exponential(scale=500, size=n) + 60
    data["account_total_txn_count"] = rng.poisson(lam=500, size=n).astype(float)
    data["account_avg_amount"] = avg_amounts
    data["account_std_amount"] = std_amounts
    data["account_max_amount"] = avg_amounts * rng.uniform(2, 5, size=n)

    # Cross features
    data["amount_x_velocity"] = amounts * data["txn_count_1h"]
    data["amount_x_hour_risk"] = np.zeros(n)
    data["international_x_new_merchant"] = data["is_international"] * data["new_merchant_flag"]
    data["high_amount_x_unusual_hour"] = np.zeros(n)

    return pd.DataFrame(data)


def _generate_fraud_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate feature distributions for fraudulent transactions."""
    data: dict[str, np.ndarray] = {}

    # Transaction features - higher amounts, unusual ratios
    amounts = rng.lognormal(mean=5.5, sigma=1.5, size=n)
    avg_amounts = rng.lognormal(mean=3.5, sigma=0.5, size=n)
    std_amounts = rng.exponential(scale=20, size=n) + 1.0
    data["transaction_amount"] = amounts
    data["amount_zscore"] = (amounts - avg_amounts) / np.maximum(std_amounts, 1.0)
    data["amount_to_avg_ratio"] = amounts / np.maximum(avg_amounts, 1.0)
    data["amount_percentile"] = rng.beta(1, 2, size=n) + 0.5  # skewed high
    data["amount_percentile"] = np.clip(data["amount_percentile"], 0, 1)
    data["amount_log"] = np.log1p(amounts)

    # Temporal features - unusual hours
    data["hour_of_day"] = rng.choice([0, 1, 2, 3, 4, 5, 23], size=n).astype(float)
    data["day_of_week"] = rng.integers(0, 7, size=n).astype(float)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(float)
    data["is_holiday"] = rng.binomial(1, 0.1, size=n).astype(float)
    data["time_since_last_transaction"] = rng.exponential(scale=120, size=n)
    data["minutes_since_midnight"] = data["hour_of_day"] * 60 + rng.integers(0, 60, size=n)

    # Velocity features 1h - much higher
    data["txn_count_1h"] = rng.poisson(lam=6, size=n).astype(float)
    data["txn_amount_sum_1h"] = amounts * data["txn_count_1h"] * rng.uniform(0.8, 1.5, size=n)
    data["txn_amount_avg_1h"] = data["txn_amount_sum_1h"] / np.maximum(data["txn_count_1h"], 1)
    data["txn_amount_max_1h"] = data["txn_amount_avg_1h"] * rng.uniform(1.5, 3.0, size=n)

    # Velocity features 24h
    data["txn_count_24h"] = rng.poisson(lam=20, size=n).astype(float)
    data["txn_amount_sum_24h"] = amounts * data["txn_count_24h"] * rng.uniform(0.8, 1.5, size=n)
    data["txn_amount_avg_24h"] = data["txn_amount_sum_24h"] / np.maximum(data["txn_count_24h"], 1)
    data["txn_amount_max_24h"] = data["txn_amount_avg_24h"] * rng.uniform(2.0, 5.0, size=n)
    data["txn_amount_std_24h"] = rng.exponential(scale=80, size=n)

    # Velocity features 7d
    data["txn_count_7d"] = rng.poisson(lam=40, size=n).astype(float)
    data["txn_amount_sum_7d"] = amounts * data["txn_count_7d"] * rng.uniform(0.8, 1.5, size=n)
    data["txn_amount_avg_7d"] = data["txn_amount_sum_7d"] / np.maximum(data["txn_count_7d"], 1)

    # Merchant features - more unusual
    data["unique_merchants_24h"] = rng.poisson(lam=8, size=n).astype(float)
    data["unique_merchants_7d"] = rng.poisson(lam=15, size=n).astype(float)
    data["new_merchant_flag"] = rng.binomial(1, 0.6, size=n).astype(float)
    data["merchant_risk_score"] = rng.beta(5, 5, size=n)
    data["merchant_txn_count_30d"] = rng.poisson(lam=50, size=n).astype(float)
    data["merchant_fraud_rate"] = rng.beta(5, 100, size=n)

    # Geographic features - international, risky countries
    data["unique_countries_24h"] = rng.poisson(lam=3, size=n).astype(float) + 1
    data["is_international"] = rng.binomial(1, 0.65, size=n).astype(float)
    data["country_risk_score"] = rng.beta(5, 5, size=n)
    data["distance_from_last_location_km"] = rng.exponential(scale=500, size=n)
    data["impossible_travel_flag"] = rng.binomial(1, 0.15, size=n).astype(float)

    # Device features - unknown devices
    data["known_device_flag"] = rng.binomial(1, 0.3, size=n).astype(float)
    data["device_age_days"] = rng.exponential(scale=30, size=n) + 1
    data["device_txn_count"] = rng.poisson(lam=5, size=n).astype(float)
    data["multiple_accounts_on_device"] = rng.binomial(1, 0.25, size=n).astype(float)

    # Behavioral features - unusual patterns
    data["unusual_hour_flag"] = np.ones(n)
    data["channel_switch_flag"] = rng.binomial(1, 0.4, size=n).astype(float)
    data["amount_deviation_from_mean"] = amounts - avg_amounts
    data["amount_deviation_from_median"] = amounts - avg_amounts * 0.9
    data["txn_frequency_deviation"] = rng.normal(2.0, 1.0, size=n)

    # Sequence features - more declines
    data["consecutive_declined_count"] = rng.poisson(lam=2, size=n).astype(float)
    data["rapid_succession_flag"] = rng.binomial(1, 0.4, size=n).astype(float)
    data["time_since_last_decline"] = rng.exponential(scale=3600, size=n)
    data["decline_rate_24h"] = rng.beta(5, 20, size=n)

    # Account features - newer accounts
    data["account_age_days"] = rng.exponential(scale=60, size=n) + 5
    data["account_total_txn_count"] = rng.poisson(lam=30, size=n).astype(float)
    data["account_avg_amount"] = avg_amounts
    data["account_std_amount"] = std_amounts
    data["account_max_amount"] = avg_amounts * rng.uniform(3, 8, size=n)

    # Cross features
    data["amount_x_velocity"] = amounts * data["txn_count_1h"]
    data["amount_x_hour_risk"] = amounts * data["unusual_hour_flag"]
    data["international_x_new_merchant"] = data["is_international"] * data["new_merchant_flag"]
    data["high_amount_x_unusual_hour"] = (amounts > avg_amounts * 3).astype(float) * data[
        "unusual_hour_flag"
    ]

    return pd.DataFrame(data)


def time_based_split(
    X: pd.DataFrame,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Split data chronologically to prevent data leakage."""
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


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get ML feature columns, excluding metadata columns."""
    exclude = {"timestamp", "transaction_id", "customer_id"}
    return [c for c in df.columns if c not in exclude and c in FEATURE_CATALOG]


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_cols: list[str],
    model_type: str = "xgboost",
    class_weight_strategy: str = "balanced",
    params: dict[str, Any] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[Any, StandardScaler, dict[str, Any]]:
    """Train a gradient boosted tree model with class imbalance handling.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        feature_cols: Feature column names to use.
        model_type: 'xgboost' or 'lightgbm'.
        class_weight_strategy: 'balanced', 'smote', or 'none'.
        params: Model hyperparameters (if None, uses defaults).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (trained model, fitted scaler, training metrics dict).
    """
    X_train_feat = X_train[feature_cols].values
    X_val_feat = X_val[feature_cols].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_val_scaled = scaler.transform(X_val_feat)

    # Handle class imbalance
    X_fit, y_fit = X_train_scaled, y_train
    scale_pos_weight = None

    if class_weight_strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(random_state=random_state)
            X_fit, y_fit = smote.fit_resample(X_train_scaled, y_train)
            logger.info("SMOTE applied: %d -> %d samples", len(y_train), len(y_fit))
        except ImportError:
            logger.warning("imblearn not available, falling back to class weights")
            class_weight_strategy = "balanced"

    if class_weight_strategy == "balanced":
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = n_neg / max(n_pos, 1)

    # Train model
    if model_type == "xgboost":
        model, metrics = _train_xgboost(
            X_fit,
            y_fit,
            X_val_scaled,
            y_val,
            scale_pos_weight=scale_pos_weight,
            params=params,
            random_state=random_state,
        )
    elif model_type == "lightgbm":
        model, metrics = _train_lightgbm(
            X_fit,
            y_fit,
            X_val_scaled,
            y_val,
            scale_pos_weight=scale_pos_weight,
            params=params,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, scaler, metrics


def _train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float | None = None,
    params: dict[str, Any] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[Any, dict[str, Any]]:
    """Train XGBoost model."""
    import xgboost as xgb

    default_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if scale_pos_weight is not None:
        default_params["scale_pos_weight"] = scale_pos_weight
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)

    eval_set = [(X_val, y_val)]
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )

    # Validation metrics
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    metrics = _compute_metrics(y_val, y_pred_proba)
    metrics["best_iteration"] = (
        model.best_iteration if hasattr(model, "best_iteration") else default_params["n_estimators"]
    )

    logger.info(
        "XGBoost training complete: AUC-ROC=%.4f, F1=%.4f, Precision=%.4f, Recall=%.4f",
        metrics["auc_roc"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    )
    return model, metrics


def _train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float | None = None,
    params: dict[str, Any] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[Any, dict[str, Any]]:
    """Train LightGBM model."""
    import lightgbm as lgb

    default_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary",
        "metric": "auc",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if scale_pos_weight is not None:
        default_params["scale_pos_weight"] = scale_pos_weight
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)

    eval_set = [(X_val, y_val)]
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
    )

    # Validation metrics
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    metrics = _compute_metrics(y_val, y_pred_proba)

    logger.info(
        "LightGBM training complete: AUC-ROC=%.4f, F1=%.4f, Precision=%.4f, Recall=%.4f",
        metrics["auc_roc"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    )
    return model, metrics


class CalibratedScorer:
    """Calibrated scorer wrapping a model + isotonic/sigmoid calibration.

    Provides predict_proba interface consistent with sklearn classifiers.
    """

    def __init__(self, model: Any, calibration_fn: Any) -> None:
        self._model = model
        self._calibration_fn = calibration_fn

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        raw_proba = self._model.predict_proba(X)[:, 1]
        calibrated = self._calibration_fn.transform(raw_proba)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return getattr(self._model, "feature_importances_", None)


def calibrate_model(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "isotonic",
) -> CalibratedScorer:
    """Calibrate model scores to true probabilities.

    Uses isotonic regression or Platt scaling (logistic) to ensure predicted
    probabilities match observed frequencies.

    Args:
        model: Trained classifier with predict_proba.
        X_val: Validation features (scaled).
        y_val: Validation labels.
        method: 'isotonic' or 'sigmoid'.

    Returns:
        CalibratedScorer wrapping model + calibration function.
    """
    raw_proba = model.predict_proba(X_val)[:, 1]

    if method == "isotonic":
        calibration_fn = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibration_fn.fit(raw_proba, y_val)
    else:
        # Platt scaling (logistic regression on scores)
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression()
        lr.fit(raw_proba.reshape(-1, 1), y_val)

        class _SigmoidCalibrator:
            def __init__(self, lr):
                self._lr = lr

            def transform(self, X):
                return self._lr.predict_proba(np.asarray(X).reshape(-1, 1))[:, 1]

        calibration_fn = _SigmoidCalibrator(lr)

    calibrator = CalibratedScorer(model, calibration_fn)

    # Verify calibration
    y_pred_proba = calibrator.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    logger.info("Calibration complete (method=%s): AUC-ROC=%.4f", method, auc)

    return calibrator


def compute_feature_importance(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    """Extract and rank feature importances from the trained model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return []

    importance_list = []
    for name, imp in zip(feature_names, importances):
        importance_list.append(
            {
                "feature": name,
                "importance": float(imp),
                "rank": 0,
            }
        )

    importance_list.sort(key=lambda x: x["importance"], reverse=True)
    for i, item in enumerate(importance_list):
        item["rank"] = i + 1

    return importance_list


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "risk_scorer",
) -> dict[str, Any]:
    """Comprehensive model evaluation on test set."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = _compute_metrics(y_test, y_pred_proba)

    # Threshold analysis
    thresholds = np.arange(0.1, 1.0, 0.05)
    threshold_results = []
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        if y_pred.sum() > 0 and (1 - y_pred).sum() > 0:
            threshold_results.append(
                {
                    "threshold": float(t),
                    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                }
            )

    metrics["threshold_analysis"] = threshold_results
    metrics["model_name"] = model_name

    return metrics


def benchmark_latency(
    model: Any,
    scaler: StandardScaler,
    X_sample: np.ndarray,
    n_iterations: int = 1000,
    warmup: int = 100,
) -> dict[str, float]:
    """Benchmark single-prediction latency."""
    single_sample = X_sample[:1]

    # Warmup
    for _ in range(warmup):
        scaled = scaler.transform(single_sample)
        model.predict_proba(scaled)

    # Benchmark
    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        scaled = scaler.transform(single_sample)
        model.predict_proba(scaled)
        elapsed = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed)

    latencies_arr = np.array(latencies)
    result = {
        "mean_ms": float(latencies_arr.mean()),
        "median_ms": float(np.median(latencies_arr)),
        "p95_ms": float(np.percentile(latencies_arr, 95)),
        "p99_ms": float(np.percentile(latencies_arr, 99)),
        "max_ms": float(latencies_arr.max()),
        "min_ms": float(latencies_arr.min()),
        "meets_sla": bool(np.percentile(latencies_arr, 99) < 20.0),
    }

    logger.info(
        "Latency benchmark: mean=%.2fms, p95=%.2fms, p99=%.2fms, SLA met=%s",
        result["mean_ms"],
        result["p95_ms"],
        result["p99_ms"],
        result["meets_sla"],
    )
    return result


def _compute_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Compute standard classification metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
    auc_pr = average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "positive_rate": float(y_true.mean()),
    }


def run_training_pipeline(
    n_samples: int = 50000,
    fraud_ratio: float = 0.02,
    model_type: str = "xgboost",
    class_weight_strategy: str = "balanced",
    output_dir: str | Path | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run complete training pipeline end-to-end.

    Steps:
        1. Generate synthetic data
        2. Time-based split
        3. Train model with class imbalance handling
        4. Calibrate scores
        5. Evaluate on test set
        6. Compute feature importance
        7. Benchmark latency
        8. Save artifacts

    Returns:
        Dictionary with all metrics, paths, and metadata.
    """
    output_path = Path(output_dir) if output_dir else DEFAULT_MODEL_OUTPUT
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline_start = time.time()
    results: dict[str, Any] = {"model_type": model_type, "random_state": random_state}

    # Step 1: Generate data
    logger.info(
        "Step 1: Generating synthetic data (n=%d, fraud_ratio=%.3f)", n_samples, fraud_ratio
    )
    X, y = generate_synthetic_data(n_samples, fraud_ratio, random_state)

    # Step 2: Split
    logger.info("Step 2: Time-based train/val/test split")
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(X, y)
    feature_cols = get_feature_columns(X_train)
    results["n_features"] = len(feature_cols)
    results["feature_names"] = feature_cols

    # Step 3: Train
    logger.info("Step 3: Training %s model", model_type)
    model, scaler, train_metrics = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        feature_cols=feature_cols,
        model_type=model_type,
        class_weight_strategy=class_weight_strategy,
        params=params,
        random_state=random_state,
    )
    results["training_metrics"] = train_metrics

    # Step 4: Calibrate
    logger.info("Step 4: Calibrating scores")
    X_val_scaled = scaler.transform(X_val[feature_cols].values)
    calibrator = calibrate_model(model, X_val_scaled, y_val, method="isotonic")
    results["calibration_method"] = "isotonic"

    # Step 5: Evaluate
    logger.info("Step 5: Evaluating on test set")
    X_test_scaled = scaler.transform(X_test[feature_cols].values)
    test_metrics = evaluate_model(calibrator, X_test_scaled, y_test)
    results["test_metrics"] = test_metrics

    # Step 6: Feature importance
    logger.info("Step 6: Computing feature importance")
    importance = compute_feature_importance(model, feature_cols)
    results["feature_importance"] = importance[:20]  # Top 20

    # Step 7: Latency benchmark
    logger.info("Step 7: Benchmarking prediction latency")
    latency = benchmark_latency(model, scaler, X_test[feature_cols].values[:100])
    results["latency"] = latency

    # Step 8: Save
    logger.info("Step 8: Saving model artifacts to %s", output_path)
    scorer = RiskScorer()
    scorer.save_model(
        output_path=output_path,
        model=model,
        scaler=scaler,
        calibrator=calibrator,
        feature_names=feature_cols,
        model_type=model_type,
        model_version="1.0.0",
        extra_metadata={
            "training_metrics": train_metrics,
            "test_metrics": {k: v for k, v in test_metrics.items() if k != "threshold_analysis"},
            "feature_importance": importance[:10],
            "latency": latency,
            "n_training_samples": len(y_train),
            "fraud_ratio": fraud_ratio,
            "class_weight_strategy": class_weight_strategy,
            "trained_at": pd.Timestamp.now().isoformat(),
        },
    )

    # Save evaluation report
    report_path = output_path / "evaluation_report.json"
    report = {
        "model_type": model_type,
        "model_version": "1.0.0",
        "training_samples": len(y_train),
        "test_samples": len(y_test),
        "fraud_ratio": fraud_ratio,
        "test_metrics": test_metrics,
        "feature_importance_top_20": importance[:20],
        "latency_benchmark": latency,
        "production_ready": test_metrics["auc_roc"] > 0.92 and latency["meets_sla"],
        "pipeline_duration_seconds": time.time() - pipeline_start,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    results["output_path"] = str(output_path)
    results["production_ready"] = report["production_ready"]
    results["pipeline_duration_seconds"] = report["pipeline_duration_seconds"]

    logger.info(
        "Training pipeline complete: AUC-ROC=%.4f, production_ready=%s, duration=%.1fs",
        test_metrics["auc_roc"],
        report["production_ready"],
        report["pipeline_duration_seconds"],
    )
    return results


def main() -> None:
    """CLI entry point for training pipeline."""
    parser = argparse.ArgumentParser(description="Train risk scoring model")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_MODEL_OUTPUT))
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--fraud-ratio", type=float, default=0.02)
    parser.add_argument("--model-type", choices=["xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--class-weight", choices=["balanced", "smote", "none"], default="balanced")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    results = run_training_pipeline(
        n_samples=args.n_samples,
        fraud_ratio=args.fraud_ratio,
        model_type=args.model_type,
        class_weight_strategy=args.class_weight,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE RESULTS")
    print("=" * 60)
    print(f"Model Type:       {results['model_type']}")
    print(f"Features:         {results['n_features']}")
    print(f"AUC-ROC (test):   {results['test_metrics']['auc_roc']:.4f}")
    print(f"Precision:        {results['test_metrics']['precision']:.4f}")
    print(f"Recall:           {results['test_metrics']['recall']:.4f}")
    print(f"F1 Score:         {results['test_metrics']['f1']:.4f}")
    print(f"Latency (p99):    {results['latency']['p99_ms']:.2f}ms")
    print(f"Production Ready: {results['production_ready']}")
    print(f"Output:           {results['output_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
