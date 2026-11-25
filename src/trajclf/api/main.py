"""FastAPI service: send a ride, receive a prediction.

Swagger UI: http://localhost:8000/docs
Prometheus metrics: http://localhost:8000/metrics
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from trajclf import __version__
from trajclf.predictor import load_predictor_from_env

log = logging.getLogger("trajclf.api")

REQUESTS = Counter("api_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram(
    "api_request_latency_seconds", "End-to-end request latency", ["endpoint"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREPROCESS = Histogram(
    "model_preprocess_seconds", "Feature computation time",
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
INFERENCE = Histogram(
    "model_inference_seconds", "Model forward time (preprocessing excluded)",
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
PREDICTIONS = Counter("model_predictions_total", "Predicted classes", ["label"])
CONFIDENCE = Histogram("model_confidence", "Top-1 probability", buckets=[i / 10 for i in range(11)])
TRAJ_LEN = Histogram("input_trajectory_points", "Raw IMU readings per ride (drift signal)",
                     buckets=(100, 500, 1000, 1800, 2500, 3500, 4800, 10000))
MODEL_INFO = Gauge("model_info", "Loaded model", ["version", "runtime"])

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor, version = load_predictor_from_env()
    STATE.update(predictor=predictor, version=version)
    MODEL_INFO.labels(version=version, runtime=predictor.runtime).set(1)
    # warm-up so the first user request is not penalised
    predictor.predict([np.random.default_rng(0).normal(size=(100, len(predictor.pre.feature_columns)))])
    log.info("model version %s loaded with runtime %s", version, predictor.runtime)
    yield
    STATE.clear()


app = FastAPI(
    title="Trajectory Classification API",
    description="Send the IMU readings of one ride, receive its class: "
                "Safe, Safe overall, Bad weather, Tandem or Reckless.",
    version=__version__,
    lifespan=lifespan,
)


class Ride(BaseModel):
    points: list[list[float]] = Field(
        ..., min_length=2,
        description="IMU readings of one ride ordered by time_index; each reading = [ax, ay, az, rx, ry, rz] "
                    "(order given by GET /model). Real rides have 1,807–4,767 readings.",
        examples=[[[0.12, -0.05, 9.81, 0.01, 0.00, -0.02],
                   [0.30, -0.10, 9.75, 0.03, 0.01, -0.01],
                   [0.25, 0.02, 9.90, 0.02, -0.01, 0.00]]],
    )


class BatchRides(BaseModel):
    rides: list[Ride] = Field(..., min_length=1, max_length=256)


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    if path != "/metrics":
        LATENCY.labels(endpoint=path).observe(time.perf_counter() - start)
        REQUESTS.labels(endpoint=path, status=response.status_code).inc()
    return response


def _run(rides: list[Ride]) -> list[Prediction]:
    predictor = STATE["predictor"]
    n_feat = len(predictor.pre.feature_columns)
    arrays = []
    for r in rides:
        a = np.asarray(r.points, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != n_feat:
            raise HTTPException(422, f"each point must have {n_feat} values: {predictor.pre.feature_columns}")
        TRAJ_LEN.observe(len(a))
        arrays.append(a)
    t0 = time.perf_counter()
    x = predictor.pre.transform_batch(arrays)
    t1 = time.perf_counter()
    z = predictor.logits(x)
    INFERENCE.observe(time.perf_counter() - t1)
    PREPROCESS.observe(t1 - t0)
    results = predictor.postprocess(z)
    out = []
    for res in results:
        PREDICTIONS.labels(label=res["label"]).inc()
        CONFIDENCE.observe(res["confidence"])
        out.append(Prediction(**res, model_version=STATE["version"]))
    return out


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok" if "predictor" in STATE else "loading"}


@app.get("/model", tags=["ops"])
def model_info() -> dict:
    p = STATE["predictor"]
    return {"version": STATE["version"], "runtime": p.runtime, "features": p.pre.feature_columns,
            "mode": p.pre.mode, "input_shape": list(p.pre.input_shape), "labels": p.labels,
            "test_macro_f1": p.meta.get("test_macro_f1")}


@app.post("/predict", response_model=Prediction, tags=["inference"])
def predict(ride: Ride) -> Prediction:
    return _run([ride])[0]


@app.post("/predict/batch", response_model=list[Prediction], tags=["inference"])
def predict_batch(batch: BatchRides) -> list[Prediction]:
    return _run(batch.rides)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
