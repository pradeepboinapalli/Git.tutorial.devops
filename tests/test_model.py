import numpy as np
from heart_disease.features import NUMERIC_FEATURES, build_logistic_regression_pipeline
from sklearn.datasets import make_classification


def _synthetic_frame(n=120):
    import pandas as pd

    X, y = make_classification(
        n_samples=n,
        n_features=len(NUMERIC_FEATURES),
        n_informative=8,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=NUMERIC_FEATURES)
    df["target"] = y
    return df


def test_pipeline_fit_predict():
    df = _synthetic_frame()
    X = df[NUMERIC_FEATURES]
    y = df["target"]
    pipe = build_logistic_regression_pipeline()
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
