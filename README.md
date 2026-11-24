# Ride classification — MLOps pipeline + AI Infrastructure

Classifies e-scooter/bike **rides** (IMU readings: accelerometer `ax ay az`, gyroscope `rx ry rz`)
into 5 classes: **Safe, Safe overall, Bad weather, Tandem, Reckless** (GBG500 dataset, 500 rides).

One repository, two courses:
* **MLOps** — preprocessing, training, MLflow tracking + registry, model storage on S3 (MinIO),
  FastAPI, web app, Docker, Kubernetes, GitHub Actions CI/CD, retraining pipeline, monitoring.
* **AI Infrastructure** — resource sizing, inference benchmark, ONNX/INT8 optimization,
  distributed-training decision, scaling/cost/rollback plan → [`ai-infrastructure/`](ai-infrastructure/README.md).

Team: *Name 1, Name 2, Name 3, Name 4, Name 5*

## Architecture

![current system](ai-infrastructure/architecture/current-system.png)

| Step | Tool | Code |
|---|---|---|
| Data extraction & checks | pandas / pyarrow | `research/inspect_data.py`, `research/check_rides.py` |
| Preprocessing | duplicates averaged, NaN-aware stats, stratified ride split (80/20, seed 42) | `src/trajclf/data.py` |
| Model | PyTorch — `mlp` (30 aggregate features, default) or `cnn` (1024×6 sequence) | `src/trajclf/model.py`, `configs/*.yaml` |
| Tracking & registry | MLflow; best model gets alias `@champion` (macro-F1 gate) | `src/trajclf/train.py` |
| Model storage | MinIO (S3) as MLflow artifact store; bundle = TorchScript + ONNX + preprocessing | `src/trajclf/bundle.py` |
| API | FastAPI + Swagger (`/docs`), Prometheus metrics (`/metrics`) | `src/trajclf/api/main.py` |
| Web app | Streamlit: upload a ride CSV, see signals + prediction | `webapp/app.py` |
| Containers | Docker images api / webapp / train | `Dockerfile.*`, `docker-compose.yml` |
| Production | Docker Desktop Kubernetes: Deployment, Service, HPA, CronJob | `k8s/` |
| CI/CD | GitHub Actions: lint + tests → images on Docker Hub → deploy (self-hosted runner) with auto-rollback | `.github/workflows/` |
| Retraining pipeline (bonus) | data → train → register → promote if better → redeploy | `.github/workflows/retrain.yml`, `pipelines/dags/`, `scripts/retrain.sh` |
| Monitoring (bonus) | Prometheus + Grafana dashboard (latency, errors, class mix, confidence, input drift) | `monitoring/` |

## Technical choices

* **Supervised learning** on GBG500 labels (confirmed with the teacher, see `docs/FINDINGS.md`);
  `SRV3490` (crash data) is not used for the 5-class task.
* **Imbalance** (Safe 416 vs Reckless 6): class-weighted cross-entropy; **macro-F1** as the main
  metric; accuracy alone is misleading (always predicting Safe gives 0.83).
* **Same held-out test split as the research phase** (100 rides), so MLflow runs compare directly
  with `docs/FINDINGS.md`: always-Safe macro-F1 0.18, random forest 0.36, first MLP 0.23
  (exploratory). The pipeline also logs a random-guess baseline to check "better than random".
* **Preprocessing inside the model bundle** (`preprocess.json`): the API and training apply the
  same transformation, so there is no training/serving skew.
* **TorchScript + ONNX export** in every bundle: the API needs no model class, and the runtime can
  be switched (`MODEL_RUNTIME=torch|onnx|onnx-int8`) for the AI-infra optimization.

## Quick start (dev)

```bash
pip install -r requirements-dev.txt
# put Project.zip content in data/raw/  (GBG500.parquet, GBG500_labels.csv) — git-ignored
docker compose up -d minio minio-init mlflow     # MinIO :9001, MLflow :5000
make data train                                  # preprocess, train, log, register, promote
docker compose up -d                             # API :8000/docs, web app :8501, Grafana :3000
python scripts/export_sample_rides.py            # test rides as CSV for the web app
```

No data? `make synthetic` generates fake rides with the same schema (for tests only).
Sequence model: `TRAJCLF_CONFIG=configs/config.cnn.yaml make data train`.

## Production (local Kubernetes)

```bash
kubectl apply -k k8s/                            # MinIO, MLflow, API (x2 + HPA), web app, retrain CronJob
kubectl -n trajclf port-forward svc/mlflow 5000 & kubectl -n trajclf port-forward svc/minio 9000 &
make data train                                  # registers @champion in the cluster's MLflow
kubectl -n trajclf rollout restart deploy/trajclf-api
helm install monitoring prometheus-community/kube-prometheus-stack -n monitoring --create-namespace -f k8s/monitoring-values.yaml
```

The CD workflow runs these steps on a self-hosted runner. Configure repository secrets:
`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `MINIO_USER`, `MINIO_PASSWORD`.

## Tests

```bash
make test                     # unit + integration (synthetic data, no server needed)
API_URL=http://localhost:8000 pytest -m e2e   # against a running API
```

## Docker images

* `docker.io/<user>/trajclf-api`
* `docker.io/<user>/trajclf-webapp`
* `docker.io/<user>/trajclf-train`

## Results

*Add screenshots from the production environment: Swagger, MLflow runs/registry, Grafana, `kubectl get pods,hpa`.*

## Repository layout

```text
src/trajclf/          package: data, model, train, bundle, predictor, api/
configs/              config.yaml (MLP) · config.cnn.yaml (CNN)
research/             first exploration scripts + saved outputs (RF and MLP baselines)
docs/                 FINDINGS.md, PROJECT_OVERVIEW.md, data.txt
webapp/  k8s/  monitoring/  pipelines/dags/  scripts/  tests/
ai-infrastructure/    sizing, benchmarks, optimization, distributed-training, cost-resilience
```
