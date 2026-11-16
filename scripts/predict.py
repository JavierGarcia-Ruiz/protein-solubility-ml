"""Inference script to predict protein solubility from raw amino acid sequence."""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from solubility.data.features import extract_single_sequence_features
from solubility.models.factory import get_feature_cols
from solubility.utils import load_config


def predict_sequence(
    sequence: str,
    config_path: str = "configs/default.yaml",
) -> dict[str, float]:
    cfg = load_config(config_path)
    models_dir = Path(cfg["paths"]["models"])
    
    # Extract features
    raw_feats = extract_single_sequence_features(sequence)
    df_feats = pd.DataFrame([raw_feats])
    
    # Load classifiers
    results = {}
    clf_path = models_dir / "deepsol_lightgbm.joblib"
    if clf_path.exists():
        clf = joblib.load(clf_path)
        feat_cols = get_feature_cols(df_feats.columns.tolist())
        prob = float(clf.predict_proba(df_feats[feat_cols].values)[:, 1][0])
        results["deepsol_solubility_probability"] = prob
        results["predicted_class"] = "Soluble" if prob >= 0.5 else "Insoluble"

    # Load regressor
    reg_path = models_dir / "esol_lightgbm.joblib"
    if reg_path.exists():
        reg = joblib.load(reg_path)
        feat_cols = get_feature_cols(df_feats.columns.tolist())
        pred_val = float(reg.predict(df_feats[feat_cols].values)[0])
        results["esol_predicted_solubility_ratio"] = pred_val

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict solubility of a protein sequence")
    parser.add_argument("--sequence", type=str, required=True, help="Amino acid sequence")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    
    res = predict_sequence(args.sequence, config_path=args.config)
    print("\n--- Protein Solubility Prediction ---")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

