"""Generate all publication-grade figures and SHAP explanations for protein-solubility-ml."""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
import umap

from solubility.models.factory import get_feature_cols
from solubility.utils import get_logger, load_config, seed_everything
from solubility.visualization.plotting import (
    CLASS_COLORS,
    MODEL_COLORS,
    MODEL_LABELS,
    OKABE_ITO,
    panel_label,
    save,
    set_style,
)


def generate_all_figures(config_path: str = "configs/default.yaml") -> None:
    set_style()
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    seed_everything(seed)
    logger = get_logger("solubility.figures", log_file=cfg["paths"]["logs"] / "figures.log")

    proc_dir = Path(cfg["paths"]["processed"])
    results_dir = Path(cfg["paths"]["results"])
    figures_dir = Path(cfg["paths"]["figures"])
    models_dir = Path(cfg["paths"]["models"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    df_train = pd.read_parquet(proc_dir / "deepsol_train.parquet")
    df_val = pd.read_parquet(proc_dir / "deepsol_val.parquet")
    df_test = pd.read_parquet(proc_dir / "deepsol_test.parquet")
    df_esol_te = pd.read_parquet(proc_dir / "esol_test.parquet")

    feature_cols = get_feature_cols(df_test.columns.tolist())
    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values.astype(int)

    # Load predictions and metrics
    preds_deepsol = pd.read_parquet(results_dir / "predictions_deepsol.parquet")
    preds_esol = pd.read_parquet(results_dir / "predictions_esol.parquet")
    metrics_clf = pd.read_csv(results_dir / "metrics_classification.csv")
    metrics_reg = pd.read_csv(results_dir / "metrics_regression.csv")

    # =========================================================================
    # Figure 1: Dataset Overview & Physicochemical Distributions
    # =========================================================================
    logger.info("Generating Figure 1: Dataset Overview...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1A: Length distributions
    ax = axes[0, 0]
    sns.kdeplot(df_train["length"], label=f"DeepSol Train (N={len(df_train):,})", color=OKABE_ITO[0], ax=ax, clip=(0, 1500), common_norm=False)
    sns.kdeplot(df_test["length"], label=f"DeepSol Test (N={len(df_test):,})", color=OKABE_ITO[1], ax=ax, clip=(0, 1500), common_norm=False)
    sns.kdeplot(df_esol_te["length"], label=f"eSOL Test (N={len(df_esol_te):,})", color=OKABE_ITO[2], ax=ax, clip=(0, 1500), common_norm=False)
    ax.set_xlabel("Sequence Length (amino acids)")
    ax.set_ylabel("Density")
    ax.set_title("Sequence Length Distributions")
    ax.legend(loc="upper right")
    panel_label(ax, "A")

    # 1B: Class balance
    ax = axes[0, 1]
    splits_data = [
        ("Train", df_train["label"].mean()),
        ("Val", df_val["label"].mean()),
        ("Test", df_test["label"].mean()),
    ]
    x_pos = np.arange(len(splits_data))
    sol_fracs = [d[1] for d in splits_data]
    insol_fracs = [1 - d[1] for d in splits_data]
    ax.bar(x_pos, sol_fracs, width=0.5, label="Soluble", color=CLASS_COLORS["Soluble"])
    ax.bar(x_pos, insol_fracs, bottom=sol_fracs, width=0.5, label="Insoluble", color=CLASS_COLORS["Insoluble"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d[0] for d in splits_data])
    ax.set_ylabel("Proportion")
    ax.set_title("Class Balance Across DeepSol Partitions")
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    panel_label(ax, "B")

    # 1C: Net charge distribution
    ax = axes[1, 0]
    sns.kdeplot(data=df_test, x="net_charge_ph7", hue="label", palette=[CLASS_COLORS["Insoluble"], CLASS_COLORS["Soluble"]], ax=ax, common_norm=False)
    ax.set_xlabel("Theoretical Net Charge at pH 7.0 (e)")
    ax.set_ylabel("Density")
    ax.set_title("Net Charge vs Solubility (DeepSol Test)")
    ax.axvline(0, color="gray", linestyle=":")
    panel_label(ax, "C")

    # 1D: GRAVY distribution
    ax = axes[1, 1]
    sns.kdeplot(data=df_test, x="gravy", hue="label", palette=[CLASS_COLORS["Insoluble"], CLASS_COLORS["Soluble"]], ax=ax, common_norm=False)
    ax.set_xlabel("Grand Average of Hydropathicity (GRAVY)")
    ax.set_ylabel("Density")
    ax.set_title("Hydrophobicity vs Solubility (DeepSol Test)")
    ax.axvline(0, color="gray", linestyle=":")
    panel_label(ax, "D")

    fig.tight_layout()
    save(fig, figures_dir / "01_dataset_overview.png")

    # =========================================================================
    # Figure 2: Model Comparison (Classification & Regression)
    # =========================================================================
    logger.info("Generating Figure 2: Model Comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 2A: DeepSol Classification (Accuracy and ROC-AUC)
    ax = axes[0]
    x = np.arange(len(metrics_clf))
    width = 0.35
    accs = metrics_clf["test_accuracy"].values
    acc_err = [
        accs - metrics_clf["test_accuracy_ci_lower"].values,
        metrics_clf["test_accuracy_ci_upper"].values - accs,
    ]
    rocs = metrics_clf["test_roc_auc"].values
    roc_err = [
        rocs - metrics_clf["test_roc_auc_ci_lower"].values,
        metrics_clf["test_roc_auc_ci_upper"].values - rocs,
    ]
    
    ax.bar(x - width/2, accs, width, yerr=acc_err, capsize=4, label="Test Accuracy", color=OKABE_ITO[0])
    ax.bar(x + width/2, rocs, width, yerr=roc_err, capsize=4, label="Test ROC-AUC", color=OKABE_ITO[1])
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in metrics_clf["model"]], rotation=25, ha="right")
    ax.set_ylim(0.45, 0.88)
    ax.set_ylabel("Score")
    ax.set_title("DeepSol Test Performance (N=2,001)")
    ax.legend(loc="lower right")
    panel_label(ax, "A")

    # 2B: eSOL Regression (R2 and Pearson r)
    ax = axes[1]
    x_reg = np.arange(len(metrics_reg))
    r2s = np.maximum(metrics_reg["test_r2"].values, -0.05)
    pearsons = metrics_reg["test_pearson_r"].values
    
    ax.bar(x_reg - width/2, r2s, width, label="Test R²", color=OKABE_ITO[2])
    ax.bar(x_reg + width/2, pearsons, width, label="Test Pearson r", color=OKABE_ITO[4])
    ax.set_xticks(x_reg)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in metrics_reg["model"]], rotation=25, ha="right")
    ax.set_ylim(-0.05, 0.75)
    ax.set_ylabel("Score")
    ax.set_title("eSOL Test Performance (N=783)")
    ax.legend(loc="upper left")
    panel_label(ax, "B")

    fig.tight_layout()
    save(fig, figures_dir / "02_model_comparison_cv_test.png")

    # =========================================================================
    # Figure 3: ROC & PR Curves
    # =========================================================================
    logger.info("Generating Figure 3: ROC & PR Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax_roc = axes[0]
    ax_pr = axes[1]

    for model_name in metrics_clf["model"]:
        col = f"{model_name}_prob"
        if col not in preds_deepsol.columns:
            continue
        prob = preds_deepsol[col].values
        
        # ROC
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_val = metrics_clf.loc[metrics_clf["model"] == model_name, "test_roc_auc"].values[0]
        color = MODEL_COLORS.get(model_name, "#555555")
        label = f"{MODEL_LABELS.get(model_name, model_name)} (AUC={roc_val:.3f})"
        ax_roc.plot(fpr, tpr, label=label, color=color, lw=2)

        # PR
        prec, rec, _ = precision_recall_curve(y_test, prob)
        pr_val = metrics_clf.loc[metrics_clf["model"] == model_name, "test_pr_auc"].values[0]
        ax_pr.plot(rec, prec, label=f"{MODEL_LABELS.get(model_name, model_name)} (AP={pr_val:.3f})", color=color, lw=2)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (0.500)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic (ROC)")
    ax_roc.legend(loc="lower right")
    panel_label(ax_roc, "A")

    baseline_prev = y_test.mean()
    ax_pr.axhline(baseline_prev, color="k", linestyle="--", alpha=0.4, label=f"Prevalence ({baseline_prev:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve (PR)")
    ax_pr.legend(loc="lower left")
    panel_label(ax_pr, "B")

    fig.tight_layout()
    save(fig, figures_dir / "03_roc_pr_curves.png")

    # =========================================================================
    # Figure 4: Calibration Curves & Confusion Matrix
    # =========================================================================
    logger.info("Generating Figure 4: Calibration & Confusion Matrix...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 4A: Reliability diagrams
    ax = axes[0]
    for model_name in ["logreg", "random_forest", "xgboost", "lightgbm"]:
        col = f"{model_name}_prob"
        if col not in preds_deepsol.columns:
            continue
        prob = preds_deepsol[col].values
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
        brier = metrics_clf.loc[metrics_clf["model"] == model_name, "test_brier"].values[0]
        ax.plot(mean_pred, frac_pos, "s-", label=f"{MODEL_LABELS[model_name]} (Brier={brier:.3f})", color=MODEL_COLORS[model_name], lw=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Fraction of Positives")
    ax.set_title("Reliability Diagram (Calibration)")
    ax.legend(loc="upper left")
    panel_label(ax, "A")

    # 4B: Confusion Matrix for best model (LightGBM)
    ax = axes[1]
    best_prob = preds_deepsol["lightgbm_prob"].values
    best_pred = (best_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, best_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    annot = np.array([
        [f"{cm[0, 0]}\n({cm_norm[0, 0]:.1%})", f"{cm[0, 1]}\n({cm_norm[0, 1]:.1%})"],
        [f"{cm[1, 0]}\n({cm_norm[1, 0]:.1%})", f"{cm[1, 1]}\n({cm_norm[1, 1]:.1%})"],
    ])

    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", cbar=True, ax=ax,
                xticklabels=["Insoluble", "Soluble"], yticklabels=["Insoluble", "Soluble"])
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("LightGBM Confusion Matrix (DeepSol Test)")
    panel_label(ax, "B")

    fig.tight_layout()
    save(fig, figures_dir / "04_calibration_confusion.png")

    # =========================================================================
    # Figure 5 & 6: SHAP Summary & Dependence Plots
    # =========================================================================
    logger.info("Computing SHAP values on DeepSol Test Set...")
    best_model_path = models_dir / "deepsol_lightgbm.joblib"
    if best_model_path.exists():
        lgb_model = joblib.load(best_model_path)
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_matrix = shap_values[1]  # positive class
        elif shap_values.ndim == 3:
            shap_matrix = shap_values[:, :, 1]
        else:
            shap_matrix = shap_values

        # 5A & 5B: SHAP Beeswarm + Feature Importance Bar
        fig = plt.figure(figsize=(13, 6))
        ax_bar = fig.add_subplot(1, 2, 1)
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        top_idx = np.argsort(mean_abs_shap)[-15:]
        y_pos = np.arange(len(top_idx))
        ax_bar.barh(y_pos, mean_abs_shap[top_idx], color=OKABE_ITO[0])
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([feature_cols[i] for i in top_idx])
        ax_bar.set_xlabel("Mean |SHAP Value| (Impact on Model Output)")
        ax_bar.set_title("Top 15 Informative Features (LightGBM)")
        panel_label(ax_bar, "A")

        # 5B: Beeswarm
        ax_bee = fig.add_subplot(1, 2, 2)
        plt.sca(ax_bee)
        shap.summary_plot(shap_matrix[:, top_idx], X_test[:, top_idx], feature_names=[feature_cols[i] for i in top_idx], show=False)
        ax_bee.set_title("SHAP Beeswarm Distribution")
        panel_label(ax_bee, "B")

        fig.tight_layout()
        save(fig, figures_dir / "05_shap_summary.png")

        # Figure 6: SHAP Dependence Plots
        logger.info("Generating Figure 6: SHAP Dependence Plots...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        dep_features = [
            ("net_charge_ph7", "Theoretical Net Charge at pH 7.0 (e)"),
            ("frac_K", "Fraction of Lysine (K)"),
            ("gravy", "Grand Average of Hydropathicity (GRAVY)"),
            ("lys_arg_ratio", "Lysine to Arginine Ratio (K / (K + R))"),
        ]
        
        for idx, (f_name, label_str) in enumerate(dep_features):
            ax = axes[idx // 2, idx % 2]
            if f_name in feature_cols:
                f_idx = feature_cols.index(f_name)
                f_vals = X_test[:, f_idx]
                s_vals = shap_matrix[:, f_idx]
                
                scatter = ax.scatter(f_vals, s_vals, c=X_test[:, feature_cols.index("net_charge_ph7")],
                                     cmap="coolwarm", s=14, alpha=0.6)
                ax.axhline(0, color="k", linestyle="--", alpha=0.3)
                ax.set_xlabel(label_str)
                ax.set_ylabel("SHAP Value (Solubility)")
                ax.set_title(f"SHAP Dependence: {f_name}")
                fig.colorbar(scatter, ax=ax, label="Net Charge (pH 7)")
                panel_label(ax, chr(ord("A") + idx))

        fig.tight_layout()
        save(fig, figures_dir / "06_shap_dependence.png")

    # =========================================================================
    # Figure 7: eSOL Regression Parity & Residuals
    # =========================================================================
    logger.info("Generating Figure 7: eSOL Parity & Residuals...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    y_esol_true = preds_esol["true_solubility"].values
    best_reg_name = "lightgbm" if "lightgbm_pred" in preds_esol.columns else "xgboost"
    y_esol_pred = preds_esol[f"{best_reg_name}_pred"].values
    reg_m = metrics_reg.loc[metrics_reg["model"] == best_reg_name].iloc[0]

    # 7A: Parity Plot
    ax = axes[0]
    ax.scatter(y_esol_true, y_esol_pred, color=OKABE_ITO[0], alpha=0.5, s=20, edgecolors="none")
    ax.plot([0, 1], [0, 1], "r--", lw=2, label="Identity line (y = x)")
    
    # Linear trendline
    slope, intercept = np.polyfit(y_esol_true, y_esol_pred, 1)
    x_vals = np.linspace(0, 1, 100)
    ax.plot(x_vals, slope * x_vals + intercept, color="navy", lw=1.5,
            label=f"Fit (R²={reg_m['test_r2']:.3f}, r={reg_m['test_pearson_r']:.3f})")

    ax.set_xlabel("Experimental eSOL Solubility")
    ax.set_ylabel("Predicted Solubility (LightGBM)")
    ax.set_title(f"eSOL Test Set Parity Plot (N={len(y_esol_true)})")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    panel_label(ax, "A")

    # 7B: Residuals Plot
    ax = axes[1]
    residuals = y_esol_true - y_esol_pred
    ax.scatter(y_esol_pred, residuals, color=OKABE_ITO[1], alpha=0.5, s=20, edgecolors="none")
    ax.axhline(0, color="r", linestyle="--", lw=2)
    ax.set_xlabel("Predicted Solubility")
    ax.set_ylabel("Residual (True - Predicted)")
    ax.set_title(f"Residual Plot (RMSE={reg_m['test_rmse']:.3f}, MAE={reg_m['test_mae']:.3f})")
    panel_label(ax, "B")

    fig.tight_layout()
    save(fig, figures_dir / "07_esol_regression_parity.png")

    # =========================================================================
    # Figure 8: MANDATORY Latent-Space UMAP Projection
    # =========================================================================
    logger.info("Computing UMAP 2D Latent Representation on DeepSol Test Set...")
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.3, metric="cosine", random_state=seed)
    embedding = reducer.fit_transform(X_test)

    # Compute silhouette score and 5-NN separation
    from sklearn.model_selection import cross_val_score
    sil = silhouette_score(embedding, y_test)
    knn_acc = float(cross_val_score(KNeighborsClassifier(n_neighbors=5), embedding, y_test, cv=5).mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 8A: UMAP coloured by true solubility
    ax = axes[0]
    idx_insol = (y_test == 0)
    idx_sol = (y_test == 1)

    ax.scatter(embedding[idx_insol, 0], embedding[idx_insol, 1],
               color=CLASS_COLORS["Insoluble"], label=f"Insoluble (N={idx_insol.sum()})",
               s=14, alpha=0.6, edgecolors="none")
    ax.scatter(embedding[idx_sol, 0], embedding[idx_sol, 1],
               color=CLASS_COLORS["Soluble"], label=f"Soluble (N={idx_sol.sum()})",
               s=14, alpha=0.6, edgecolors="none")

    # Annotate class centroids with clean vertical offsets so labels never collide
    c_insol = embedding[idx_insol].mean(axis=0)
    c_sol = embedding[idx_sol].mean(axis=0)
    ax.scatter(*c_insol, color="black", marker="X", s=130, edgecolors="white", linewidths=1.5, zorder=10)
    ax.annotate(
        "Insoluble Centroid",
        xy=(c_insol[0], c_insol[1]),
        xytext=(-18, -22),
        textcoords="offset points",
        ha="right",
        va="top",
        fontweight="bold",
        fontsize=9.5,
        arrowprops=dict(arrowstyle="-", color="#333333", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#333333", alpha=0.92),
        zorder=11,
    )

    ax.scatter(*c_sol, color="black", marker="X", s=130, edgecolors="white", linewidths=1.5, zorder=10)
    ax.annotate(
        "Soluble Centroid",
        xy=(c_sol[0], c_sol[1]),
        xytext=(16, 16),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=9.5,
        arrowprops=dict(arrowstyle="-", color="#333333", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#333333", alpha=0.92),
        zorder=11,
    )

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title(f"UMAP of Feature Space (DeepSol Test, N=2,001)\nSilhouette: {sil:.3f} | 5-CV 5-NN Separation: {knn_acc:.1%}")
    ax.legend(loc="upper right")
    panel_label(ax, "A")

    # 8B: UMAP coloured by continuous Net Charge (biological driver)
    ax = axes[1]
    net_charge_test = df_test["net_charge_ph7"].values
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=net_charge_test,
                         cmap="coolwarm", s=14, alpha=0.7, edgecolors="none", vmin=-25, vmax=25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Theoretical Net Charge at pH 7.0 (e)")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title("Latent Space Manifold Coloured by Net Charge\nNegative Charge Drives Solubility Cluster Separation")
    panel_label(ax, "B")

    fig.tight_layout()
    save(fig, figures_dir / "08_latent_space_umap.png")
    logger.info("All 8 publication figures generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate figures for protein-solubility-ml")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    generate_all_figures(config_path=args.config)

