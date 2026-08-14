"""
Load and validate the raw Telco CSV.
"""
import pandas as pd
from src.config import DATA_RAW, TARGET, CUSTOMER_ID


def load_raw() -> pd.DataFrame:
    """Load raw CSV and perform basic validation."""
    df = pd.read_csv(DATA_RAW)

    # TotalCharges has blank strings — coerce to float, NaN filled downstream
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop customerID (not a feature)
    df = df.drop(columns=[CUSTOMER_ID], errors="ignore")

    # Encode target: Yes→1, No→0
    df[TARGET] = (df[TARGET].str.strip().str.lower() == "yes").astype(int)

    print(f"[data_loader] Loaded {len(df)} rows × {df.shape[1]} columns.")
    print(f"[data_loader] Churn rate: {df[TARGET].mean():.2%}")
    return df


def split_X_y(df: pd.DataFrame):
    """Separate features from target."""
    from src.config import TARGET
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y
