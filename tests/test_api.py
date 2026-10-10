import heart_disease.api.app as app_module
import joblib
import pytest
from fastapi.testclient import TestClient
from heart_disease.api.app import app, load_model
from heart_disease.features import NUMERIC_FEATURES, build_logistic_regression_pipeline


@pytest.fixture
def client(tmp_path, monkeypatch):
    import pandas as pd
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=80,
        n_features=len(NUMERIC_FEATURES),
        n_informative=6,
        random_state=1,
    )
    df = pd.DataFrame(X, columns=NUMERIC_FEATURES)
    pipe = build_logistic_regression_pipeline()
    pipe.fit(df, y)
    model_path = tmp_path / "best_model.pkl"
    joblib.dump(pipe, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setattr(app_module, "_model", None)
    load_model()

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict(client):
    payload = {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 1,
        "ca": 0,
        "thal": 2,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["confidence"] <= 1
    assert "probabilities" in body
