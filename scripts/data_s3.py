"""Move the raw dataset between data/raw/ and the MinIO (S3) bucket used by the retraining pipeline.

    python scripts/data_s3.py upload     # once: data/raw/GBG500.* -> s3://data/gbg500/
    python scripts/data_s3.py download   # in the pipeline: s3://data/gbg500/ -> data/raw/

Env: MLFLOW_S3_ENDPOINT_URL (default http://localhost:9000), AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
(default minioadmin / minioadmin, the demo credentials of k8s/secrets.yaml).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3

BUCKET, PREFIX = "data", "gbg500"
FILES = ("GBG500.parquet", "GBG500_labels.csv")
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"upload", "download"}:
        sys.exit(__doc__)
    s3 = client()
    if sys.argv[1] == "upload":
        if BUCKET not in [b["Name"] for b in s3.list_buckets()["Buckets"]]:
            s3.create_bucket(Bucket=BUCKET)
        for name in FILES:
            s3.upload_file(str(RAW / name), BUCKET, f"{PREFIX}/{name}")
            print(f"uploaded {name} -> s3://{BUCKET}/{PREFIX}/{name}")
    else:
        RAW.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            s3.download_file(BUCKET, f"{PREFIX}/{name}", str(RAW / name))
            print(f"downloaded s3://{BUCKET}/{PREFIX}/{name} -> {RAW / name}")


if __name__ == "__main__":
    main()
