export PYTHONPATH := src
export MLFLOW_TRACKING_URI ?= http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL ?= http://localhost:9000
export AWS_ACCESS_KEY_ID ?= minioadmin
export AWS_SECRET_ACCESS_KEY ?= minioadmin

.PHONY: install data synthetic train train-local api webapp test lint up down k8s bench

install:          ## dev dependencies
	pip install -r requirements-dev.txt
data:             ## preprocess data/raw/GBG500.* -> data/processed
	python -m trajclf.data
synthetic:        ## generate synthetic data instead of the real one
	python -m trajclf.data --synthetic
train:            ## train + log to MLflow + register + promote
	python -m trajclf.train
train-local:      ## train without MLflow, bundle in artifacts/bundle
	python -m trajclf.train --no-mlflow
api:              ## run API locally on the local bundle
	MODEL_DIR=artifacts/bundle uvicorn trajclf.api.main:app --reload
webapp:
	API_URL=http://localhost:8000 streamlit run webapp/app.py
test:
	pytest -m "not e2e" --cov=trajclf
lint:
	ruff check src tests ai-infrastructure
up:
	docker compose up -d --build
down:
	docker compose down
k8s:              ## deploy on the current kube context (Docker Desktop)
	kubectl apply -k k8s/
bench:            ## AI-infra baseline + optimisation benchmark on the local bundle
	python ai-infrastructure/optimization/quantize.py --bundle artifacts/bundle
	python ai-infrastructure/benchmarks/benchmark.py --bundle artifacts/bundle --runtimes torch onnx onnx-int8
