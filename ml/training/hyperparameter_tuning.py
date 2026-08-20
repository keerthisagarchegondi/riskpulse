"""Optuna-based hyperparameter optimization for risk scoring models.

Performs Bayesian optimization to find the best hyperparameters
for XGBoost/LightGBM fraud detection models with time-series aware
cross-validation.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.train_risk_scorer import (  # noqa: E402
    DEFAULT_MODEL_OUTPUT,
    DEFAULT_RANDOM_STATE,
    generate_synthetic_data,
    get_feature_columns,
    time_based_split,
)

logger = logging.getLogger(__name__)


def create_xgboost_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """Create Optuna objective function for XGBoost hyperparameter tuning.

    Returns:
        Callable objective for optuna.study.optimize().
    """
    import optuna

    def objective(trial: optuna.Trial) -> float:
        import xgboost as xgb

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        return auc

    return objective


def create_lightgbm_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """Create Optuna objective function for LightGBM hyperparameter tuning.

    Returns:
        Callable objective for optuna.study.optimize().
    """
    import optuna

    def objective(trial: optuna.Trial) -> float:
        import lightgbm as lgb

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "objective": "binary",
            "metric": "auc",
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }

        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
        )

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        return auc

    return objective


def create_cv_objective(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "xgboost",
    n_splits: int = 5,
    scale_pos_weight: float | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """Create cross-validated Optuna objective with time-series splits.

    More robust than single train/val split but slower.
    """
    import optuna

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        if model_type == "xgboost":
            import xgboost as xgb

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": random_state,
                "n_jobs": -1,
                "tree_method": "hist",
            }
            if scale_pos_weight is not None:
                params["scale_pos_weight"] = scale_pos_weight
            model_cls = xgb.XGBClassifier
        else:
            import lightgbm as lgb

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "num_leaves": trial.suggest_int("num_leaves", 15, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "objective": "binary",
                "metric": "auc",
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": -1,
            }
            if scale_pos_weight is not None:
                params["scale_pos_weight"] = scale_pos_weight
            model_cls = lgb.LGBMClassifier

        auc_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            model = model_cls(**params)
            if model_type == "xgboost":
                model.fit(
                    X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False
                )
            else:
                model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)])

            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]

            if len(np.unique(y_fold_val)) > 1:
                fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
                auc_scores.append(fold_auc)

            # Pruning: report intermediate result
            trial.report(np.mean(auc_scores) if auc_scores else 0.0, len(auc_scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(auc_scores) if auc_scores else 0.0

    return objective


def run_optimization(
    n_trials: int = 100,
    n_samples: int = 30000,
    fraud_ratio: float = 0.02,
    model_type: str = "xgboost",
    use_cv: bool = False,
    n_cv_splits: int = 5,
    output_dir: str | Path | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Run full hyperparameter optimization pipeline.

    Args:
        n_trials: Number of Optuna trials to run.
        n_samples: Number of synthetic samples to generate.
        fraud_ratio: Fraud ratio in synthetic data.
        model_type: 'xgboost' or 'lightgbm'.
        use_cv: Use cross-validated objective (slower but more robust).
        n_cv_splits: Number of CV folds if use_cv=True.
        output_dir: Directory to save results.
        random_state: Seed for reproducibility.
        timeout: Maximum time in seconds for optimization.

    Returns:
        Dict with best params, best score, and study summary.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    output_path = Path(output_dir) if output_dir else DEFAULT_MODEL_OUTPUT
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic data for hyperparameter tuning...")
    X, y = generate_synthetic_data(n_samples, fraud_ratio, random_state)

    feature_cols = get_feature_columns(X)
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(X, y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[feature_cols].values)
    X_val_scaled = scaler.transform(X_val[feature_cols].values)
    X_test_scaled = scaler.transform(X_test[feature_cols].values)

    # Compute class weight
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / max(n_pos, 1)

    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"risk_scorer_{model_type}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
    )

    # Create objective
    if use_cv:
        X_combined = np.vstack([X_train_scaled, X_val_scaled])
        y_combined = np.concatenate([y_train, y_val])
        objective = create_cv_objective(
            X_combined,
            y_combined,
            model_type=model_type,
            n_splits=n_cv_splits,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
        )
    else:
        if model_type == "xgboost":
            objective = create_xgboost_objective(
                X_train_scaled,
                y_train,
                X_val_scaled,
                y_val,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
            )
        else:
            objective = create_lightgbm_objective(
                X_train_scaled,
                y_train,
                X_val_scaled,
                y_val,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
            )

    # Run optimization
    start_time = time.time()
    logger.info("Starting Optuna optimization: %d trials, model=%s", n_trials, model_type)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    elapsed = time.time() - start_time

    # Results
    best_params = study.best_params
    best_score = study.best_value

    # Evaluate best params on test set
    test_auc = _evaluate_best_on_test(
        best_params,
        model_type,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )

    results = {
        "model_type": model_type,
        "best_params": best_params,
        "best_val_auc": float(best_score),
        "test_auc": float(test_auc),
        "n_trials": len(study.trials),
        "n_completed": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "optimization_time_seconds": elapsed,
        "use_cv": use_cv,
    }

    # Save results
    results_path = output_path / "hyperparameter_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(
        "Optimization complete: best_val_auc=%.4f, test_auc=%.4f, trials=%d, time=%.1fs",
        best_score,
        test_auc,
        len(study.trials),
        elapsed,
    )

    return results


def _evaluate_best_on_test(
    params: dict[str, Any],
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale_pos_weight: float | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> float:
    """Retrain with best params and evaluate on test set."""
    full_params = {**params, "random_state": random_state, "n_jobs": -1}
    if scale_pos_weight is not None:
        full_params["scale_pos_weight"] = scale_pos_weight

    if model_type == "xgboost":
        import xgboost as xgb

        full_params.update(
            {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist"}
        )
        model = xgb.XGBClassifier(**full_params)
        model.fit(X_train, y_train, verbose=False)
    else:
        import lightgbm as lgb

        full_params.update({"objective": "binary", "metric": "auc", "verbosity": -1})
        model = lgb.LGBMClassifier(**full_params)
        model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred_proba)


def main() -> None:
    """CLI entry point for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for risk scorer")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=30000)
    parser.add_argument("--fraud-ratio", type=float, default=0.02)
    parser.add_argument("--model-type", choices=["xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--use-cv", action="store_true")
    parser.add_argument("--n-cv-splits", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_MODEL_OUTPUT))
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--timeout", type=int, default=None, help="Max seconds for optimization")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    results = run_optimization(
        n_trials=args.n_trials,
        n_samples=args.n_samples,
        fraud_ratio=args.fraud_ratio,
        model_type=args.model_type,
        use_cv=args.use_cv,
        n_cv_splits=args.n_cv_splits,
        output_dir=args.output_dir,
        random_state=args.random_state,
        timeout=args.timeout,
    )

    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Model Type:      {results['model_type']}")
    print(f"Best Val AUC:    {results['best_val_auc']:.4f}")
    print(f"Test AUC:        {results['test_auc']:.4f}")
    print(f"Trials:          {results['n_completed']} completed, {results['n_pruned']} pruned")
    print(f"Duration:        {results['optimization_time_seconds']:.1f}s")
    print(f"\nBest Parameters:")
    for k, v in results["best_params"].items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
