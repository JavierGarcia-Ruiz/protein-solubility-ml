"""Comprehensive evaluation metrics for classification and regression with bootstrapping."""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics from ground truth and predicted probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["pr_auc"] = float("nan")

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics: MSE, RMSE, MAE, R2, Pearson r, Spearman rho."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Pearson
    if np.std(y_pred) > 1e-9 and np.std(y_true) > 1e-9:
        pearson_r, _ = stats.pearsonr(y_true, y_pred)
        spearman_rho, _ = stats.spearmanr(y_true, y_pred)
    else:
        pearson_r = float("nan")
        spearman_rho = float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred_or_prob: np.ndarray,
    task: str = "classification",
    metric_name: str = "roc_auc",
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap point estimate and percentile confidence interval [ci_lower, ci_upper]."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    estimates: list[float] = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample_true = y_true[idx]
        sample_pred = y_pred_or_prob[idx]
        
        if task == "classification":
            if len(np.unique(sample_true)) < 2:
                continue
            m = compute_classification_metrics(sample_true, sample_pred)
        else:
            m = compute_regression_metrics(sample_true, sample_pred)
            
        val = m.get(metric_name, float("nan"))
        if not np.isnan(val):
            estimates.append(val)

    if not estimates:
        return float("nan"), float("nan"), float("nan")

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(estimates, alpha * 100))
    upper = float(np.percentile(estimates, (1.0 - alpha) * 100))
    median = float(np.median(estimates))
    return median, lower, upper

