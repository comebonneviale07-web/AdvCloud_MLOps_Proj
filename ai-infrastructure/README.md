# AI Infrastructure contribution

Extends the trajectory-classification MLOps project (assignment: [`ASSIGNMENT.md`](ASSIGNMENT.md)).
**Rule: measure, do not claim.** Each number below must come from a file in this folder.
Anything still marked *to measure* has not been measured.

| # | Deliverable | Where | Status |
|---|---|---|---|
| 1 | Current architecture + resource inventory | [`architecture/current-system.png`](architecture/current-system.png), [`resource-sizing.md`](resource-sizing.md) | **done (measured)** |
| 2 | Baseline benchmark | `benchmarks/benchmark.py`, `load_test.py` → [`benchmarks/results/`](benchmarks/results/) | **done (measured)**: model-level + HTTP end-to-end |
| 3 | Optimization (ONNX Runtime FP32 + static INT8) | `optimization/quantize.py`, same benchmark | **done (measured)**: ONNX FP32 retained |
| 4 | Distributed-training decision | [`distributed-training/technical-analysis.md`](distributed-training/technical-analysis.md) | **done**: no DDP/FSDP/ZeRO (1.55 s CPU training, 68 kB model states) |
| 5 | Scaling, monitoring, cost, rollback | [`cost-resilience/infrastructure-plan.md`](cost-resilience/infrastructure-plan.md), [`architecture/proposed-system.png`](architecture/proposed-system.png) | design + cost ≈ $90/month · HPA and rollback *to test* |

## Step 1 — Inventory

| Component | Implementation |
|---|---|
| Model / framework | PyTorch. `mlp`: RideMLP, 30 aggregate features, 4,229 parameters (team's first model, docs/FINDINGS.md). `cnn`: 1D-CNN on 1024 × 6 resampled readings, 94,405 parameters |
| Preprocessing | `src/trajclf/data.py` (duplicates averaged, NaN-aware statistics, stratified 80/20 split, seed 42) |
| Training / tracking / registry | `src/trajclf/train.py` → MLflow (params, metrics, hardware tags, sizes) + Registry alias `@champion` |
| Model storage | MinIO (S3) via MLflow artifacts: `bundle/` = TorchScript + ONNX + preprocess.json |
| Prediction API | FastAPI `src/trajclf/api/main.py` (Swagger `/docs`, Prometheus `/metrics`) |
| Docker / K8s | `Dockerfile.*`, `docker-compose.yml`, `k8s/` (Deployment, HPA, CronJob) |
| CI/CD | `.github/workflows/ci.yml`, `cd.yml` (self-hosted runner, auto-rollback), `retrain.yml` |
| Monitoring | Prometheus + Grafana (`monitoring/`) |
| Where it runs | training: laptop CPU or Colab T4 · serving: Docker Desktop Kubernetes (CPU) |

## Step 3 — Baseline results (measured 24/11/2026)

Conditions: model RideMLP (4,229 parameters), bundle `artifacts/bundle` trained with `configs/config.yaml`
(seed 42, identical config and test metrics to MLflow model version 1); TorchScript runtime; CPU only;
**1 intra-op thread**; 20 warm-up + 300 timed calls per batch size; each runtime in its own process.
Raw data: `benchmarks/results/raw-latencies.csv`; environment: `benchmarks/results/environment.json`.

| Metric | Value | Measurement conditions |
|---|---|---|
| Model and version | RideMLP (PyTorch 2.15, TorchScript) | MLflow `trajectory-classifier` v1 config |
| Hardware | Intel Core i7-13620H (16 logical CPUs), 34 GB RAM, no GPU | Windows 11 laptop |
| Model size (MB) | 0.030 (TorchScript) · 0.018 (ONNX) | serialized files |
| Mean latency (ms) | **0.021** (batch 1) · 0.022 (batch 8) · 0.025 (batch 32) | 300 calls, 1 thread |
| p95 latency (ms) | **0.027** (batch 1) · 0.030 · 0.032 | same |
| Throughput (rides/s, in-process) | ≈48,500 (batch 1) · ≈1.27 M (batch 32) | batch size × 1000 / mean latency |
| Throughput (requests/s, HTTP) | **48.8** (1 client) · **94.2** (4 clients, peak) · 81.7 (16 clients) | `load_test.py`, 20 s per level, 1 API container (Docker, 1 uvicorn worker), real rides ≈3,000 readings, 0 errors |
| Memory usage (MB) | 331 (process RSS after load, mostly the PyTorch library) | psutil |
| Preprocessing per ride (ms) | **0.71 mean · 0.94 p95** | 100 real test rides, ≈3,000 readings each |
| Classification quality | macro-F1 **0.317**, accuracy 0.80 | 100 held-out test rides (random guess 0.069, always-Safe 0.18) |
| Training | 1.55 s, 100 epochs, batch 32, CPU, peak RSS 406 MB | train.py |

### End-to-end HTTP load test (measured, Docker Compose, 1 API container, TorchScript)

| Concurrent clients | Requests (20 s) | Errors | Throughput (req/s) | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 977 | 0 | 48.8 | 20.5 | 15.5 | 56.5 | 118.8 |
| 4 | 1,891 | 0 | **94.2** | 42.4 | 25.0 | 128.4 | 155.7 |
| 16 | 1,640 | 0 | 81.7 | 195.6 | 206.2 | 288.8 | 374.7 |

**Interpretation**
* **The model is < 0.1 % of the request time.** Mean request latency is 20.5 ms at 1 client, while the
  model forward pass takes 0.021 ms and preprocessing 0.71 ms. The remaining ≈19.8 ms is HTTP, JSON
  parsing and Pydantic validation of ≈18,000 floats per ride (≈100 kB payload).
* **One container saturates at ≈94 req/s.** From 4 to 16 clients throughput falls (94 → 82 req/s)
  and latency explodes (p50 25 → 206 ms): requests queue behind a single uvicorn worker process.
  This is the capacity figure used for scaling (see `cost-resilience/infrastructure-plan.md`).
* **Consequences for optimization:** ONNX speeds up the 0.02 ms model step but cannot change
  end-to-end latency much. Bigger wins: (1) more workers/pods (horizontal scaling), (2) a lighter
  payload (send the 30 features, or a binary/compressed format), (3) skipping per-float Pydantic
  validation for large arrays.
* Limits: client and server on the same laptop (they compete for CPU); one run per level.

## Step 4 — Optimization: ONNX Runtime (FP32) and static INT8 quantization

**Hypothesis (set before testing):** ONNX Runtime's graph optimizations lower latency vs TorchScript
on CPU at batch 1; INT8 further shrinks the file ~4× with a macro-F1 drop < 0.02.
**Fair conditions:** same weights, same test split, same batch sizes, same thread count and repetitions,
same machine and session, one process per runtime. INT8 calibrated on 200 *training* rides.

| Metric (batch 1) | Baseline TorchScript | ONNX FP32 | ONNX INT8 | Interpretation |
|---|---|---|---|---|
| Mean latency (ms) | 0.021 | **0.009** | 0.015 | ONNX FP32 is 2.3× faster |
| p95 latency (ms) | 0.027 | **0.009** | 0.029 | FP32 also has the most stable tail |
| Throughput (rides/s) | 48,511 | **113,486** | 65,895 | ×2.3 for FP32 |
| Memory RSS (MB) | 331 | **183** | 183 | −45 %: ONNX Runtime is much lighter than the PyTorch library |
| Model file (MB) | 0.030 | 0.018 | **0.009** | INT8 halves the ONNX file |
| Macro-F1 (test) | 0.317 | 0.317 | 0.313 | FP32 identical; INT8 −0.004 |

**Conclusions**
1. **ONNX Runtime FP32 is retained** for serving (`MODEL_RUNTIME=onnx`): 2.3× lower latency, 45 % less memory, no quality loss.
2. **INT8 is slower than FP32** (hypothesis rejected for latency). With 4,229 parameters, the
   quantize/dequantize nodes cost more than the INT8 arithmetic saves; INT8 pays off on large,
   memory-bandwidth-bound models, not here.
3. **The model is not the bottleneck.** Computing the 30 statistics of one ride takes 0.71 ms,
   ≈80× the ONNX forward pass (0.009 ms); JSON parsing of a ~100 kB request will add more. Further
   latency work should target preprocessing and payload size, not the network.
4. **Limits:** one laptop, one session, sub-0.1 ms timings are sensitive to OS scheduling;
   the test set has only 1 Reckless ride, so per-class quality is not established.

## How to reproduce (≈15 min on Colab)

```bash
# data: unzip Project.zip so that data/raw/GBG500.parquet and data/raw/GBG500_labels.csv exist
pip install -r requirements.txt
PYTHONPATH=src python -m trajclf.data
PYTHONPATH=src python -m trajclf.train --no-mlflow             # or with MLflow running: python -m trajclf.train
python ai-infrastructure/sizing.py                              # -> resource-sizing.md
python ai-infrastructure/optimization/quantize.py               # -> artifacts/bundle/model.int8.onnx
python ai-infrastructure/benchmarks/benchmark.py --runtimes torch onnx onnx-int8
# API-level, against docker compose or the K8s service:
python ai-infrastructure/benchmarks/load_test.py --url http://localhost:8000 --concurrency 1 4 16 --tag torch-2pods
python ai-infrastructure/architecture/diagrams.py               # regenerate the PNG diagrams
```

## Owners and next actions

| Task | Owner | Done |
|---|---|---|
| Run benchmark.ipynb on Colab (CPU + T4), commit `benchmarks/results/` | *name* | ☐ |
| Run comparison.ipynb, write the interpretation | *name* | ☐ |
| Fill training time / GPU memory in technical-analysis.md | *name* | ☐ |
| Load test + HPA test on local K8s, Grafana screenshot | *name* | ☐ |
| Rollback drill (bad image + bad model), record time-to-recover | *name* | ☐ |
