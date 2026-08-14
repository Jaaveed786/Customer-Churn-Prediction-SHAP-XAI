"""
Central configuration — all column names, paths, and constants live here.
Import this in every other module to avoid magic strings.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "figures"

TARGET = "Churn"
CUSTOMER_ID = "customerID"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_COLS = [
    "SeniorCitizen",  # already 0/1
]

CAT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Engineered features added by feature_engineering.engineer_features()
ENGINEERED_NUM_COLS = [
    "charges_per_month_ratio",
    "num_services",
    "contract_risk",
]
ENGINEERED_CAT_COLS = [
    "tenure_group",
]
ENGINEERED_BIN_COLS = [
    "has_security_or_backup",
    "is_auto_payment",
]

ALL_NUMERIC = NUMERIC_COLS + ENGINEERED_NUM_COLS + BINARY_COLS + ENGINEERED_BIN_COLS
ALL_CATEGORICAL = CAT_COLS + ENGINEERED_CAT_COLS

# Business threshold — chosen to maximise recall (see evaluate.py)
# Rationale: missed churner ~$300 loss vs. unnecessary retention offer ~$20.
# 15:1 cost ratio justifies lowering threshold from default 0.5.
BUSINESS_THRESHOLD = 0.35

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
