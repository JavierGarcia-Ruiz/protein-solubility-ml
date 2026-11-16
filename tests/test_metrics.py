"""Unit tests for classification and regression metrics."""

import numpy as np
import pytest
from solubility.evaluation.metrics import (
    bootstrap_metric_ci,
    compute_classification_metrics,
    compute_regression_metrics,
)


def test_classification_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3])
    m = compute_classification_metrics(y_true, y_prob)

    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["f1"] == 1.0
    assert m["mcc"] == 1.0
    assert m["brier"] < 0.1


def test_regression_metrics():
    y_true = np.array([0.1, 0.2, 0.5, 0.8, 1.0])
    y_pred = np.array([0.15, 0.22, 0.48, 0.79, 0.95])
    m = compute_regression_metrics(y_true, y_pred)

    assert m["r2"] > 0.95
    assert m["rmse"] < 0.05
    assert m["pearson_r"] > 0.98


def test_bootstrap_metric_ci():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0] * 10)
    y_prob = np.array([0.8, 0.7, 0.3, 0.2, 0.9, 0.1, 0.8, 0.4] * 10)
    med, lo, hi = bootstrap_metric_ci(y_true, y_prob, task="classification", metric_name="roc_auc", n_resamples=100)

    assert 0.8 < med <= 1.0
    assert lo <= med <= hi

