"""
Unit tests for feature_engineering.py
Ensures the single source of truth function is correct and stable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import engineer_features


@pytest.fixture
def sample_customer():
    return pd.DataFrame([{
        "tenure": 12,
        "MonthlyCharges": 65.0,
        "TotalCharges": 780.0,
        "SeniorCitizen": 0,
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }])


def test_num_services_range(sample_customer):
    """num_services must be between 0 and 9."""
    df = engineer_features(sample_customer)
    assert 0 <= df["num_services"].iloc[0] <= 9


def test_contract_risk_mapping(sample_customer):
    """Month-to-month contract must map to risk=2."""
    df = engineer_features(sample_customer)
    assert df["contract_risk"].iloc[0] == 2


def test_tenure_group_labels(sample_customer):
    """tenure_group must produce one of the expected labels."""
    df = engineer_features(sample_customer)
    assert df["tenure_group"].iloc[0] in ["0-12", "13-24", "25-48", "49-72"]


def test_is_auto_payment_false_for_electronic_check(sample_customer):
    """Electronic check is NOT an automatic payment."""
    df = engineer_features(sample_customer)
    assert df["is_auto_payment"].iloc[0] == 0


def test_is_auto_payment_true_for_bank_transfer():
    """Bank transfer (automatic) IS an automatic payment."""
    row = pd.DataFrame([{"PaymentMethod": "Bank transfer (automatic)",
                         "tenure": 6, "MonthlyCharges": 50, "TotalCharges": 300,
                         "SeniorCitizen": 0, "gender": "Female",
                         "Partner": "No", "Dependents": "No",
                         "PhoneService": "No", "MultipleLines": "No phone service",
                         "InternetService": "DSL",
                         "OnlineSecurity": "Yes", "OnlineBackup": "No",
                         "DeviceProtection": "No", "TechSupport": "No",
                         "StreamingTV": "No", "StreamingMovies": "No",
                         "Contract": "One year",
                         "PaperlessBilling": "No",
                         }])
    df = engineer_features(row)
    assert df["is_auto_payment"].iloc[0] == 1


def test_engineer_features_does_not_modify_input(sample_customer):
    """Input DataFrame must not be mutated (copy is returned)."""
    original_cols = set(sample_customer.columns)
    engineer_features(sample_customer)
    assert set(sample_customer.columns) == original_cols


def test_has_security_or_backup(sample_customer):
    """OnlineBackup=Yes → has_security_or_backup=1."""
    df = engineer_features(sample_customer)
    assert df["has_security_or_backup"].iloc[0] == 1


def test_charges_per_month_ratio_nonnegative(sample_customer):
    """Ratio must be non-negative."""
    df = engineer_features(sample_customer)
    assert df["charges_per_month_ratio"].iloc[0] >= 0
