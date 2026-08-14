"""
Feature engineering — SINGLE SOURCE OF TRUTH.

This module is imported by BOTH src/train.py AND app/streamlit_app.py.
Never reimplement these transforms inline in the app; always call engineer_features().
This prevents training-serving skew.
"""
import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 6 engineered features on top of the raw columns.
    Input: raw DataFrame (after data_loader.load_raw(), before preprocessing).
    Returns: new DataFrame with extra columns appended.
    Does NOT modify the input in place.
    """
    df = df.copy()

    # 1. tenure_group — non-linear tenure effect
    # Hypothesis: new customers (0-12 mo) and long-tenured (49+) have very different churn rates.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
        right=True,
        include_lowest=True,
    ).astype(str)

    # 2. charges_per_month_ratio — detects billing anomalies
    # Hypothesis: TotalCharges / tenure should ≈ MonthlyCharges; large deviations are signals.
    df["charges_per_month_ratio"] = (
        df["TotalCharges"] / (df["tenure"] + 1)
    ).round(4)

    # 3. num_services — engagement proxy
    # Hypothesis: customers with more add-ons are more engaged and less likely to churn.
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    def _has_service(col_val):
        return 0 if str(col_val).lower() in ("no", "no internet service", "no phone service") else 1

    df["num_services"] = df[service_cols].apply(lambda col: col.map(_has_service)).sum(axis=1)

    # 4. has_security_or_backup — bundling effect
    # Hypothesis: customers with security or backup have a reason to stay (sticky service).
    df["has_security_or_backup"] = (
        (df["OnlineSecurity"].str.lower() == "yes") |
        (df["OnlineBackup"].str.lower() == "yes")
    ).astype(int)

    # 5. is_auto_payment — auto-pay customers churn less (passive inertia)
    # Hypothesis: automatic payments reduce churn by eliminating active cancellation triggers.
    auto_pay_methods = {"bank transfer (automatic)", "credit card (automatic)"}
    df["is_auto_payment"] = (
        df["PaymentMethod"].str.lower().isin(auto_pay_methods)
    ).astype(int)

    # 6. contract_risk — ordinal encoding with business logic
    # Month-to-month = highest risk; two year = lowest risk.
    contract_map = {
        "Month-to-month": 2,
        "One year": 1,
        "Two year": 0,
    }
    df["contract_risk"] = df["Contract"].map(contract_map).fillna(1).astype(int)

    return df


FEATURE_DESCRIPTIONS = {
    "tenure": "Number of months the customer has been with the company",
    "MonthlyCharges": "Current monthly bill amount ($)",
    "TotalCharges": "Total amount charged over tenure ($)",
    "tenure_group": "Tenure bucketed into 12-month bands",
    "charges_per_month_ratio": "TotalCharges ÷ (tenure+1): detects billing anomalies",
    "num_services": "Count of add-on services subscribed (0-9)",
    "has_security_or_backup": "1 if customer has OnlineSecurity OR OnlineBackup",
    "is_auto_payment": "1 if payment is automatic (bank transfer or credit card)",
    "contract_risk": "Contract riskiness: Month-to-month=2, One year=1, Two year=0",
    "Contract": "Contract type",
    "InternetService": "Internet service type",
    "PaymentMethod": "Payment method",
    "SeniorCitizen": "1 if customer is a senior citizen",
    "Partner": "Whether the customer has a partner",
    "Dependents": "Whether the customer has dependents",
}
