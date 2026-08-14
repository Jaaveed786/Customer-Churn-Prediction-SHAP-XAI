"""
Generate and save SHAP summary plots for the README and Streamlit app.
Configured with dark theme styling (#0f1117 background) so charts blend seamlessly into the UI.
Run after training: python -m src.generate_shap_plots
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

from src.config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from src.data_loader import load_raw, split_X_y
from src.feature_engineering import engineer_features
from sklearn.model_selection import train_test_split

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_all():
    print("Loading data and models for SHAP plot generation...")
    df_raw = load_raw()
    df = engineer_features(df_raw)
    X, y = split_X_y(df)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    X_test_t = preprocessor.transform(X_test)

    # Apply dark theme styling globally for matplotlib
    plt.style.use("dark_background")

    # ── 1. Random Forest SHAP Summary (Dot Plot) ──────────────────────────────
    print("Generating RF SHAP summary plot...")
    tree_explainer = joblib.load(MODELS_DIR / "shap_tree_explainer.pkl")
    shap_vals_rf = tree_explainer.shap_values(X_test_t[:300])
    shap_vals_rf_1 = shap_vals_rf[1] if isinstance(shap_vals_rf, list) else shap_vals_rf

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    
    shap.summary_plot(
        shap_vals_rf_1,
        X_test_t[:300],
        feature_names=feature_names,
        show=False,
        plot_size=None,
        color_bar=True,
    )
    plt.title("Random Forest — Global SHAP Feature Impact", color="#a78bfa", fontsize=12, pad=15, fontweight="bold")
    plt.gcf().patch.set_facecolor("#0f1117")
    plt.savefig(REPORTS_DIR / "shap_summary_rf.png", dpi=180, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  Saved -> {REPORTS_DIR / 'shap_summary_rf.png'}")

    # ── 2. Random Forest SHAP Bar Plot (Mean Absolute SHAP) ───────────────────
    print("Generating RF SHAP bar plot...")
    fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
    fig_bar.patch.set_facecolor("#0f1117")
    ax_bar.set_facecolor("#0f1117")
    
    shap.summary_plot(
        shap_vals_rf_1,
        X_test_t[:300],
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        plot_size=None,
        color="#7c3aed",
    )
    plt.title("Random Forest — Feature Importance (Mean |SHAP|)", color="#a78bfa", fontsize=12, pad=15, fontweight="bold")
    plt.gcf().patch.set_facecolor("#0f1117")
    plt.savefig(REPORTS_DIR / "shap_bar_rf.png", dpi=180, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  Saved -> {REPORTS_DIR / 'shap_bar_rf.png'}")

    # ── 3. Logistic Regression SHAP Summary ──────────────────────────────────
    print("Generating LR SHAP summary plot...")
    linear_explainer = joblib.load(MODELS_DIR / "shap_linear_explainer.pkl")
    shap_vals_lr = linear_explainer.shap_values(X_test_t[:300])

    fig_lr, ax_lr = plt.subplots(figsize=(8, 5))
    fig_lr.patch.set_facecolor("#0f1117")
    ax_lr.set_facecolor("#0f1117")

    shap.summary_plot(
        shap_vals_lr,
        X_test_t[:300],
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title("Logistic Regression — Global SHAP Feature Impact", color="#a78bfa", fontsize=12, pad=15, fontweight="bold")
    plt.gcf().patch.set_facecolor("#0f1117")
    plt.savefig(REPORTS_DIR / "shap_summary_lr.png", dpi=180, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  Saved -> {REPORTS_DIR / 'shap_summary_lr.png'}")

    print("\nAll SHAP plots regenerated with dark styling.")


if __name__ == "__main__":
    generate_all()
