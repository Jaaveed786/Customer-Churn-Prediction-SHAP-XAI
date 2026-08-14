"""
Evaluation utilities: metrics, threshold tuning, and ROC curves.
Dark-mode styling (#0f1117) enforced for all plots.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    precision_recall_curve,
    f1_score,
)

from src.config import REPORTS_DIR, BUSINESS_THRESHOLD

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test,
    model_name: str,
    threshold: float = BUSINESS_THRESHOLD,
) -> dict:
    """Compute and print full evaluation at the business threshold."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    print(f"    ROC-AUC: {auc:.4f}  |  F1 (threshold={threshold}): {f1:.4f}")

    _plot_confusion_matrix(cm, model_name)

    return {
        "roc_auc": round(auc, 4),
        "f1": round(f1, 4),
        "precision_churn": round(report["1"]["precision"], 4),
        "recall_churn": round(report["1"]["recall"], 4),
        "threshold_used": threshold,
    }


def find_optimal_threshold(
    model,
    X_test: np.ndarray,
    y_test,
    model_name: str,
) -> float:
    """Plot precision & recall vs threshold curves on dark background."""
    probs = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161b26")

    ax.plot(thresholds, precisions[:-1], label="Precision (Accuracy of Risk Flags)", color="#818cf8", linewidth=2.2)
    ax.plot(thresholds, recalls[:-1], label="Recall (% Churners Caught)", color="#34d399", linewidth=2.2)
    ax.axvline(
        x=BUSINESS_THRESHOLD,
        color="#f43f5e",
        linestyle="--",
        linewidth=2,
        label=f"Business Threshold ({BUSINESS_THRESHOLD})",
    )

    idx = np.searchsorted(thresholds, BUSINESS_THRESHOLD)
    if idx < len(thresholds):
        ax.annotate(
            f"P={precisions[idx]:.0%}\nR={recalls[idx]:.0%}",
            xy=(BUSINESS_THRESHOLD, recalls[idx]),
            xytext=(BUSINESS_THRESHOLD + 0.06, recalls[idx] - 0.15),
            fontsize=9,
            color="#ffffff",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#1e293b", ec="#f43f5e", lw=1.5),
            arrowprops=dict(arrowstyle="->", color="#f43f5e", lw=1.5),
        )

    ax.set_xlabel("Probability Threshold", fontsize=10, color="#94a3b8")
    ax.set_ylabel("Score", fontsize=10, color="#94a3b8")
    ax.set_title(
        f"Precision-Recall Trade-off — {model_name.replace('_', ' ').title()}",
        fontsize=11, fontweight="bold", color="#f8fafc", pad=12
    )
    ax.legend(fontsize=8.5, facecolor="#1e293b", edgecolor="#334155", labelcolor="#f1f5f9")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#334155")
    fig.tight_layout()

    out_path = REPORTS_DIR / f"{model_name}_threshold_analysis.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"    Threshold plot saved -> {out_path}")

    return BUSINESS_THRESHOLD


def plot_roc_curves_comparison(models: dict, X_test: np.ndarray, y_test):
    """Overlay ROC curves for all models on a dark canvas."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161b26")

    colors = ["#a78bfa", "#34d399", "#fbbf24"]

    for (name, model), color in zip(models.items(), colors):
        disp = RocCurveDisplay.from_estimator(
            model, X_test, y_test,
            name=name.replace("_", " ").title(),
            ax=ax,
        )
        disp.line_.set_color(color)
        disp.line_.set_linewidth(2.2)

    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1.2, label="Random Guess (AUC = 0.50)")
    ax.set_title("ROC Curves — Model Diagnostic Comparison", fontsize=11, fontweight="bold", color="#f8fafc", pad=12)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f1f5f9", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#334155")
    fig.tight_layout()

    out = REPORTS_DIR / "roc_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"  ROC comparison saved -> {out}")


def _plot_confusion_matrix(cm: np.ndarray, model_name: str):
    """Save a dark styled confusion matrix heatmap."""
    import seaborn as sns
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161b26")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax, cbar=False,
        annot_kws={"size": 11, "weight": "bold", "color": "#f8fafc"}
    )
    ax.set_title(f"Confusion Matrix — {model_name.replace('_', ' ').title()}", fontsize=10, color="#f8fafc", pad=10)
    ax.set_ylabel("Actual Class", fontsize=9, color="#94a3b8")
    ax.set_xlabel("Predicted Class", fontsize=9, color="#94a3b8")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    fig.tight_layout()

    out = REPORTS_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
