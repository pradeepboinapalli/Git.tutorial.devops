from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports" / "figures"
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()

RAW_FILENAME = "processed.cleveland.data"
PROCESSED_FILENAME = "heart_disease_clean.csv"

# UCI Cleveland heart disease (14 features + target)
COLUMN_NAMES = [
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
    "num",
]

RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)
