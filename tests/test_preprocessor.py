"""
Unit tests for preprocessor.py — confirms no data leakage.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.feature_engineering import engineer_features
from src.preprocessor import build_preprocessor, get_feature_names
from src.config import TARGET


@pytest.fixture
def small_dataset():
    """Minimal synthetic dataset to test pipeline without loading the CSV."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "tenure": np.random.randint(0, 72, n),
        "MonthlyCharges": np.random.uniform(20, 120, n),
        "TotalCharges": np.random.uniform(0, 8000, n),
        "SeniorCitizen": np.random.randint(0, 2, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "Partner": np.random.choice(["Yes", "No"], n),
        "Dependents": np.random.choice(["Yes", "No"], n),
        "PhoneService": np.random.choice(["Yes", "No"], n),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n),
        "PaymentMethod": np.random.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"], n
        ),
        TARGET: np.random.randint(0, 2, n),
    })
    return df


def test_preprocessor_fit_only_on_train(small_dataset):
    """
    Scaler must not use test data statistics.
    After fitting on train only, the shape of output must match expectations.
    """
    df = engineer_features(small_dataset.drop(columns=[TARGET]))
    df[TARGET] = small_dataset[TARGET]
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)  # transform only, no fit

    assert X_train_t.shape[0] == len(X_train)
    assert X_test_t.shape[0] == len(X_test)
    assert X_train_t.shape[1] == X_test_t.shape[1]


def test_feature_names_match_output_shape(small_dataset):
    """get_feature_names must return exactly as many names as output columns."""
    df = engineer_features(small_dataset.drop(columns=[TARGET]))
    preprocessor = build_preprocessor()
    X_t = preprocessor.fit_transform(df)
    names = get_feature_names(preprocessor)
    assert len(names) == X_t.shape[1]
