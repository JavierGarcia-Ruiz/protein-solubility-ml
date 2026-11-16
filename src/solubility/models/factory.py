"""Model builders and Optuna objective factories for classification and regression."""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
import lightgbm as lgb
import xgboost as xgb


def get_feature_cols(df_columns: list[str]) -> list[str]:
    """Return all feature column names, excluding metadata."""
    exclude = {"sequence", "label", "split", "gene", "solubility"}
    return [c for c in df_columns if c not in exclude]


def build_classifier(
    model_name: str,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    n_jobs: int = 6,
) -> BaseEstimator:
    """Instantiate a classification estimator by name."""
    params = params or {}
    
    if model_name == "dummy":
        return DummyClassifier(strategy="stratified", random_state=seed)
        
    elif model_name == "logreg":
        c_val = params.get("C", 0.1)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=c_val,
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            )),
        ])
        
    elif model_name == "svm_rbf":
        c_val = params.get("C", 1.0)
        gamma = params.get("gamma", "scale")
        base_svc = SVC(C=c_val, gamma=gamma, random_state=seed, max_iter=3000)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(estimator=base_svc, cv=3, ensemble=False)),
        ])
        
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 18),
            min_samples_split=params.get("min_samples_split", 4),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            max_features=params.get("max_features", "sqrt"),
            random_state=seed,
            n_jobs=n_jobs,
        )
        
    elif model_name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            eval_metric="logloss",
        )
        
    elif model_name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 400),
            num_leaves=params.get("num_leaves", 31),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
        
    else:
        raise ValueError(f"Unknown classifier model_name: {model_name}")


def build_regressor(
    model_name: str,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    n_jobs: int = 6,
) -> BaseEstimator:
    """Instantiate a regression estimator by name."""
    params = params or {}
    
    if model_name == "dummy":
        return DummyRegressor(strategy="mean")
        
    elif model_name == "ridge":
        alpha = params.get("alpha", 10.0)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=alpha, random_state=seed)),
        ])
        
    elif model_name == "svm_rbf":
        c_val = params.get("C", 1.0)
        gamma = params.get("gamma", "scale")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR(C=c_val, gamma=gamma, epsilon=0.05, max_iter=3000)),
        ])
        
    elif model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 16),
            min_samples_split=params.get("min_samples_split", 4),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            max_features=params.get("max_features", "sqrt"),
            random_state=seed,
            n_jobs=n_jobs,
        )
        
    elif model_name == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=params.get("n_estimators", 350),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.04),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
        )
        
    elif model_name == "lightgbm":
        return lgb.LGBMRegressor(
            n_estimators=params.get("n_estimators", 350),
            num_leaves=params.get("num_leaves", 31),
            learning_rate=params.get("learning_rate", 0.04),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
        
    else:
        raise ValueError(f"Unknown regressor model_name: {model_name}")
