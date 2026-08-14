"""
SHAP explainability wrapper.
Loads pre-saved explainers and returns SHAP values + a natural-language summary.
Robust to all SHAP data types (list, numpy ndarray 2D/3D, Explanation object).
"""
import numpy as np
import joblib
from src.config import MODELS_DIR


def load_explainer(model_name: str):
    """Load the pre-fitted SHAP explainer for the given model."""
    if model_name in ("random_forest", "rf"):
        return joblib.load(MODELS_DIR / "shap_tree_explainer.pkl")
    else:
        return joblib.load(MODELS_DIR / "shap_linear_explainer.pkl")


def get_shap_values(model_name: str, X_transformed: np.ndarray) -> np.ndarray:
    """
    Returns 1D SHAP values array for a single row (shape: [n_features]).
    Handles list, Explanation object, 2D (1, N) and 3D (1, N, 2) numpy arrays robustly.
    For binary classification, extracts class 1 (churn).
    """
    explainer = load_explainer(model_name)
    raw_shap = explainer.shap_values(X_transformed)

    # 1. Handle list of arrays (e.g. [class0_array, class1_array])
    if isinstance(raw_shap, list):
        if len(raw_shap) == 0:
            return np.zeros(X_transformed.shape[1])
        elif len(raw_shap) > 1:
            arr = raw_shap[1]
        else:
            arr = raw_shap[0]
    # 2. Handle Explanation object (SHAP >= 0.40)
    elif hasattr(raw_shap, "values"):
        arr = raw_shap.values
    else:
        arr = raw_shap

    arr = np.asarray(arr)

    # 3. Handle 3D array: (n_samples, n_features, n_classes) -> e.g. (1, 54, 2)
    if arr.ndim == 3:
        if arr.shape[2] > 1:
            arr = arr[0, :, 1]  # Sample 0, all features, class 1 (churn)
        else:
            arr = arr[0, :, 0]
    # 4. Handle 2D array: (n_samples, n_features) -> e.g. (1, 54)
    elif arr.ndim == 2:
        arr = arr[0, :]  # Sample 0, all features

    return arr.flatten()


def top_reasons(
    shap_vals: np.ndarray,
    feature_names: list[str],
    n: int = 3,
) -> tuple[list, list]:
    """
    Return top-n features increasing and decreasing churn risk.
    Returns: (positive_reasons, negative_reasons)
    Each element: {"feature": str, "shap": float, "direction": str}
    """
    vals = np.asarray(shap_vals).flatten()
    idx_sorted = np.argsort(np.abs(vals))[::-1]

    positive = []
    negative = []

    for idx in idx_sorted:
        feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        shap_val = float(vals[idx])
        entry = {
            "feature": feat_name,
            "shap": round(shap_val, 4),
            "direction": "increases" if shap_val > 0 else "decreases",
        }
        if shap_val > 0 and len(positive) < n:
            positive.append(entry)
        elif shap_val < 0 and len(negative) < n:
            negative.append(entry)
        if len(positive) == n and len(negative) == n:
            break

    return positive, negative


def natural_language_summary(
    prob: float,
    threshold: float,
    positive_reasons: list,
    negative_reasons: list,
    model_name: str,
) -> str:
    """Generate a 2-3 sentence plain-English explanation of the prediction."""
    risk_level = "HIGH" if prob >= threshold else "LOW"
    pct = f"{prob * 100:.0f}%"

    pos_str = ", ".join(
        f"{r['feature'].replace('_', ' ')} (+{abs(r['shap']):.2f})"
        for r in positive_reasons[:2]
    )
    neg_str = ", ".join(
        f"{r['feature'].replace('_', ' ')} (-{abs(r['shap']):.2f})"
        for r in negative_reasons[:1]
    )

    summary = (
        f"The {model_name.replace('_', ' ').title()} model assigns this customer "
        f"a {pct} churn probability — classified as {risk_level} RISK "
        f"(threshold: {threshold}). "
    )
    if pos_str:
        summary += f"Key drivers increasing churn risk: {pos_str}. "
    if neg_str:
        summary += f"Protective factors: {neg_str}."

    return summary
