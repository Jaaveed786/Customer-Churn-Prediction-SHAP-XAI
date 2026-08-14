"""
Train Logistic Regression and Random Forest, compare them, save models and explainers.

Run:
    python -m src.train
    OR
    make train
"""
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
import shap

from src.config import (
    MODELS_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS, TARGET
)
from src.data_loader import load_raw, split_X_y
from src.feature_engineering import engineer_features
from src.preprocessor import build_preprocessor, get_feature_names
from src.evaluate import (
    evaluate_model, find_optimal_threshold, plot_roc_curves_comparison
)


def train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -- 1. Load & engineer features --------------------------------------------
    print("=" * 60)
    print("STEP 1 - Loading data")
    df_raw = load_raw()
    df = engineer_features(df_raw)

    X, y = split_X_y(df)

    # -- 2. Train/Test split (stratified) ---------------------------------------
    print("\nSTEP 2 - Train/Test split (stratified, 80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"  Train churn rate: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    # -- 3. Preprocessing pipeline - fit on X_train only -----------------------
    print("\nSTEP 3 - Fitting preprocessor on X_train only (no leakage)")
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    print(f"  Feature matrix shape after preprocessing: {X_train_t.shape}")

    # -- 4. Define models -------------------------------------------------------
    print("\nSTEP 4 - Training models")
    models = {
        "logistic_regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\n  -> {name}")
        model.fit(X_train_t, y_train)

        # Cross-validation on train
        cv_scores = cross_val_score(model, X_train_t, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"    CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # Hold-out test
        metrics = evaluate_model(model, X_test_t, y_test, name)
        metrics["cv_auc_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_auc_std"] = round(cv_scores.std(), 4)

        # Threshold tuning
        threshold = find_optimal_threshold(model, X_test_t, y_test, name)
        metrics["business_threshold"] = threshold

        results[name] = metrics
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")
        print(f"    Saved -> models/{name}.pkl")

    # -- 5. SHAP explainers - fit on training data, pickle immediately ----------
    # Gap #3 fix: both explainer objects are saved so the app never needs to refit.
    print("\nSTEP 5 - Building & saving SHAP explainers")

    rf_model = models["random_forest"]
    lr_model = models["logistic_regression"]

    tree_explainer = shap.TreeExplainer(rf_model)
    # LinearExplainer needs background data — baked in here, never needed again at inference
    linear_explainer = shap.LinearExplainer(lr_model, X_train_t)

    joblib.dump(tree_explainer, MODELS_DIR / "shap_tree_explainer.pkl")
    joblib.dump(linear_explainer, MODELS_DIR / "shap_linear_explainer.pkl")
    print("  Saved -> models/shap_tree_explainer.pkl")
    print("  Saved -> models/shap_linear_explainer.pkl")

    # -- 6. Save preprocessor & feature names ----------------------------------
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    print("  Saved -> models/preprocessor.pkl")
    print("  Saved -> models/feature_names.pkl")

    # -- 7. Comparison summary --------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    for name, m in results.items():
        print(f"\n{name.upper()}")
        for k, v in m.items():
            print(f"  {k:30s}: {v}")

    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved -> models/results.json")

    plot_roc_curves_comparison(models, X_test_t, y_test)
    print("\nTraining complete!")


if __name__ == "__main__":
    train()
