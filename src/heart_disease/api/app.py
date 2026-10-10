"""FastAPI application for heart disease inference."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from heart_disease.api.schemas import PatientFeatures, PredictionResponse
from heart_disease.config import ARTIFACTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart_disease.api")

REQUEST_COUNT = Counter(
    "heart_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
PREDICT_LATENCY = Histogram(
    "heart_predict_latency_seconds",
    "Latency of /predict",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

_model = None


def _model_path() -> Path:
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        return Path(env_path)
    return ARTIFACTS / "best_model.pkl"


def load_model():
    global _model
    path = _model_path()
    if not path.exists():
        alt = ARTIFACTS / "best_model"
        if alt.exists():
            import mlflow.sklearn

            _model = mlflow.sklearn.load_model(str(alt))
            logger.info("Loaded MLflow model from %s", alt)
            return _model
        raise FileNotFoundError(f"Model not found at {path} or {alt}")
    _model = joblib.load(path)
    logger.info("Loaded joblib model from %s", path)
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except FileNotFoundError as exc:
        logger.warning("Startup without model: %s", exc)
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    description="MLOps Assignment — UCI Heart Disease classifier",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health")
def health():
    ready = _model is not None
    return {"status": "ok" if ready else "degraded", "model_loaded": ready}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures, request: Request):
    if _model is None:
        REQUEST_COUNT.labels(request.method, "/predict", "503").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    row = pd.DataFrame([features.model_dump()])
    try:
        proba = _model.predict_proba(row)[0]
        pred = int(_model.predict(row)[0])
    except Exception as exc:
        REQUEST_COUNT.labels(request.method, "/predict", "500").inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        PREDICT_LATENCY.observe(time.perf_counter() - start)

    confidence = float(proba[pred])
    label = "heart_disease" if pred == 1 else "no_heart_disease"
    REQUEST_COUNT.labels(request.method, "/predict", "200").inc()

    return PredictionResponse(
        prediction=pred,
        label=label,
        confidence=confidence,
        probabilities={
            "no_heart_disease": float(proba[0]),
            "heart_disease": float(proba[1]),
        },
    )
