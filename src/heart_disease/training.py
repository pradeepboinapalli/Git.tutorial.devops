"""Training, evaluation, and MLflow logging."""

from __future__ import annotations

import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate

from heart_disease.config import (
    ARTIFACTS,
    CV_FOLDS,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    REPORTS,
    TEST_SIZE,
)
from heart_disease.data import load_processed, train_test_split_xy
from heart_disease.features import get_model_registry


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def cross_validate_model(model, X, y) -> dict[str, float]:
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=CV_FOLDS,
        scoring=scoring,
        n_jobs=-1,
    )
    return {
        k.replace("test_", "cv_"): float(np.mean(v))
        for k, v in cv_results.items()
        if k.startswith("test_")
    }


def train_and_select(
    experiment_name: str = "heart_disease_classification",
    register_best: bool = True,
) -> tuple[str, object, dict[str, float]]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    df = load_processed()
    X_train, X_test, y_train, y_test = train_test_split_xy(df, TEST_SIZE, RANDOM_STATE)

    best_name = ""
    best_model = None
    best_auc = -1.0
    best_metrics: dict[str, float] = {}

    for name, pipeline in get_model_registry().items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("cv_folds", CV_FOLDS)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)

            cv_metrics = cross_validate_model(pipeline, X_train, y_train)
            for k, v in cv_metrics.items():
                mlflow.log_metric(k, v)

            pipeline.fit(X_train, y_train)
            test_metrics = evaluate_model(pipeline, X_test, y_test)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            if test_metrics["roc_auc"] > best_auc:
                best_auc = test_metrics["roc_auc"]
                best_name = name
                best_model = pipeline
                best_metrics = {**cv_metrics, **{f"test_{k}": v for k, v in test_metrics.items()}}

    if best_model is None:
        raise RuntimeError("No model was trained.")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    metrics_path = ARTIFACTS / "best_model_metrics.json"
    metrics_path.write_text(
        json.dumps({"model": best_name, "metrics": best_metrics}, indent=2),
        encoding="utf-8",
    )

    if register_best:
        mlflow.sklearn.save_model(best_model, str(ARTIFACTS / "best_model"))
        joblib.dump(best_model, ARTIFACTS / "best_model.pkl")

    with mlflow.start_run(run_name="best_model_summary"):
        mlflow.log_param("selected_model", best_name)
        for k, v in best_metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(best_model, artifact_path="production_model")
        if REPORTS.exists():
            mlflow.log_artifacts(str(REPORTS), artifact_path="eda_figures")

    return best_name, best_model, best_metrics
