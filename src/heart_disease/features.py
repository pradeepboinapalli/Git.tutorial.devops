"""Sklearn preprocessing and model pipelines."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Numeric features for Cleveland dataset (all columns are numeric)
NUMERIC_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def build_logistic_regression_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def get_model_registry() -> dict[str, Pipeline]:
    return {
        "logistic_regression": build_logistic_regression_pipeline(),
        "random_forest": build_random_forest_pipeline(),
    }
