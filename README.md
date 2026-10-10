# Heart Disease MLOps (Assignment 01)

End-to-end MLOps solution for the **UCI Heart Disease (Cleveland)** dataset: EDA, training with MLflow, reproducible sklearn pipelines, CI/CD, Dockerized FastAPI, Kubernetes manifests, and Prometheus monitoring.

## Repository layout

```
├── src/heart_disease/     # Package: data, features, training, API
├── scripts/               # CLI: download, EDA, train
├── notebooks/             # Jupyter workflow
├── tests/                 # Pytest unit tests
├── k8s/                   # Kubernetes deployment manifests
├── monitoring/            # Prometheus & Grafana config
├── .github/workflows/     # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
└── reports/               # EDA figures + written report
```

## Prerequisites

- Python **3.10+**
- (Optional) Docker Desktop, kubectl, Minikube
- Git

## Quick start (local)

```powershell
cd c:\Users\saromishra\Desktop\MTech\MLOps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"

python scripts/download_data.py
python scripts/run_eda.py
python scripts/train.py
```

### MLflow UI

```powershell
mlflow ui --backend-store-uri mlruns
```

Open http://127.0.0.1:5000 and compare **logistic_regression** vs **random_forest** runs.

### Run API locally

```powershell
$env:MODEL_PATH = "models/best_model.pkl"
uvicorn heart_disease.api.app:app --reload --app-dir src --host 0.0.0.0 --port 8000
```

- Health: http://localhost:8000/health  
- Docs: http://localhost:8000/docs  
- Predict:

```powershell
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@sample_request.json"
```

## Tests & lint

```powershell
ruff check src tests scripts
pytest tests -v
```

## Docker

```powershell
docker build -t heart-disease-mlops:latest .
docker run --rm -p 8000:8000 heart-disease-mlops:latest
```

## Docker Compose (API + Prometheus + Grafana)

```powershell
docker compose up --build
```

| Service    | URL                         |
|-----------|-----------------------------|
| API       | http://localhost:8000       |
| Prometheus| http://localhost:9090     |
| Grafana   | http://localhost:3000 (admin/admin) |

Scrape metrics at http://localhost:8000/metrics and add panels in Grafana (datasource is pre-provisioned).

## Kubernetes (Minikube / Docker Desktop)

```powershell
docker build -t heart-disease-mlops:latest .
# Minikube: minikube image load heart-disease-mlops:latest

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods,svc -n heart-mlops
```

Access via LoadBalancer external IP, or port-forward:

```powershell
kubectl port-forward -n heart-mlops svc/heart-disease-api 8080:80
curl http://localhost:8080/health
```

Optional Ingress (nginx): apply `k8s/ingress.yaml` and add `heart.local` to hosts file.

## CI/CD

GitHub Actions workflow `.github/workflows/ci.yml`:

1. Ruff lint  
2. Pytest with coverage  
3. Data download + training + MLflow artifacts  
4. Docker build + smoke test on `/health` and `/predict`

Push to `main`/`master` to trigger. Upload workflow screenshots to `screenshots/` for your report.

## Assignment mapping

| Task | Implementation |
|------|----------------|
| 1 EDA | `scripts/run_eda.py`, `notebooks/01_eda_and_training.ipynb` |
| 2 Models | Logistic Regression + Random Forest, 5-fold CV, test metrics |
| 3 MLflow | `src/heart_disease/training.py`, `mlruns/` |
| 4 Packaging | `models/best_model.pkl`, sklearn Pipeline with scaler |
| 5 CI/CD | `.github/workflows/ci.yml`, `tests/` |
| 6 Docker | `Dockerfile`, FastAPI `/predict` |
| 7 K8s | `k8s/*.yaml` |
| 8 Monitoring | Request logs, Prometheus `/metrics`, Grafana compose |
| 9 Docs | `README.md`, `reports/ASSIGNMENT_REPORT.md` |

## Written report & video

- Expand `reports/ASSIGNMENT_REPORT.md` with your screenshots and export to PDF/DOCX (~10 pages).
- Record a short walkthrough: data → train → MLflow → Docker → predict → CI → (optional) K8s/monitoring.

## License

Academic project — UCI dataset used per repository terms.
"# MLOps_Assignment_" 
