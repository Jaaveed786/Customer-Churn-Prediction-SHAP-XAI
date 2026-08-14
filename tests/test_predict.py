"""
Integration test — end-to-end prediction must return probability in [0, 1].
Requires trained models. Skip if models not found.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd
import joblib
from src.feature_engineering import engineer_features
from src.config import MODELS_DIR


SAMPLE_CUSTOMER = {
    "tenure": 6,
    "MonthlyCharges": 79.5,
    "TotalCharges": 477.0,
    "SeniorCitizen": 0,
    "gender": "Female",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


@pytest.mark.skipif(
    not (MODELS_DIR / "random_forest.pkl").exists(),
    reason="Models not trained yet — run `make train` first",
)
def test_rf_prediction_is_probability():
    """Random Forest prediction must return a float in [0, 1]."""
    rf = joblib.load(MODELS_DIR / "random_forest.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    df = engineer_features(pd.DataFrame([SAMPLE_CUSTOMER]))
    X_t = preprocessor.transform(df)
    prob = rf.predict_proba(X_t)[0, 1]
    assert 0.0 <= prob <= 1.0


@pytest.mark.skipif(
    not (MODELS_DIR / "logistic_regression.pkl").exists(),
    reason="Models not trained yet — run `make train` first",
)
def test_lr_prediction_is_probability():
    """Logistic Regression prediction must return a float in [0, 1]."""
    lr = joblib.load(MODELS_DIR / "logistic_regression.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    df = engineer_features(pd.DataFrame([SAMPLE_CUSTOMER]))
    X_t = preprocessor.transform(df)
    prob = lr.predict_proba(X_t)[0, 1]
    assert 0.0 <= prob <= 1.0


@pytest.mark.skipif(
    not (MODELS_DIR / "random_forest.pkl").exists(),
    reason="Models not trained yet — run `make train` first",
)
def test_high_risk_profile_flagged():
    """
    A clearly high-risk customer (month-to-month, no security, fiber optic,
    electronic check, short tenure) should be flagged.
    """
    from src.config import BUSINESS_THRESHOLD
    rf = joblib.load(MODELS_DIR / "random_forest.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    df = engineer_features(pd.DataFrame([SAMPLE_CUSTOMER]))
    X_t = preprocessor.transform(df)
    prob = rf.predict_proba(X_t)[0, 1]
    assert prob >= BUSINESS_THRESHOLD, f"Expected high-risk customer to be flagged, got prob={prob:.3f}"
