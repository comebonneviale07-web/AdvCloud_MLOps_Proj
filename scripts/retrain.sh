#!/usr/bin/env bash
# Retraining pipeline: data -> preprocess -> train -> register -> (promote if better)
# Used by .github/workflows/retrain.yml, the Airflow DAG and the K8s Job.
set -euo pipefail
: "${MLFLOW_TRACKING_URI:?set MLFLOW_TRACKING_URI}"

# 1. Retrieve data (from MinIO bucket "data" if configured, else use local data/raw)
if [[ -n "${DATA_S3_URI:-}" ]]; then
  python - <<'PY'
import os, boto3
from urllib.parse import urlparse
u = urlparse(os.environ["DATA_S3_URI"])
s3 = boto3.client("s3", endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"))
os.makedirs("data/raw", exist_ok=True)
for key in ("GBG500.parquet", "GBG500_labels.csv"):
    s3.download_file(u.netloc, f"{u.path.strip('/')}/{key}".lstrip("/"), f"data/raw/{key}")
    print("downloaded", key)
PY
fi

# 2. Preprocess + 3. train/register/promote
python -m trajclf.data
python -m trajclf.train
