"""
Build the sklearn ColumnTransformer + Pipeline.

Key design principle: the Pipeline is fit ONLY on X_train.
It is then pickled and reused in the app — no re-fitting at inference.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import ALL_NUMERIC, ALL_CATEGORICAL


def build_preprocessor() -> ColumnTransformer:
    """
    Returns an unfitted ColumnTransformer.
    Fit it with preprocessor.fit(X_train), then transform with preprocessor.transform(X).
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # handles TotalCharges NaN
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, ALL_NUMERIC),
            ("cat", categorical_pipeline, ALL_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names after fitting, for SHAP labelling."""
    return list(preprocessor.get_feature_names_out())
