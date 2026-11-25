# Mutualised-Project-Oral-Defence

# AI Infrastructure — Project Kickoff Workshop

**Project:** Extending the existing trajectory-classification MLOps project  
**Environment:** Google Colab | GCP project  
**Work mode:** Teams

## Purpose

Continue your **existing MLOps project**. Do **not** start a new machine learning project. Add an AI Infrastructure contribution that explains the model's computing requirements, measures inference performance, investigates one realistic optimization, and designs a scalable, cost-aware system.

> **Measure, do not claim.** Distinguish measurements you actually obtained from designs, assumptions, and estimates.

## Expected AI Infrastructure contribution

1. **Resource sizing:** Document the model and framework, dataset characteristics, CPU/GPU, RAM/VRAM, model file size, and training duration.
2. **Baseline benchmarking:** Measure inference latency (mean and preferably p95), throughput, and memory usage; document test conditions.
3. **Optimization:** Choose an appropriate method (e.g., ONNX Runtime, supported quantization, or inference configuration) and compare before/after performance and prediction quality.
4. **Distributed training decision:** Assess whether DDP, FSDP, or ZeRO is justified. A well-supported decision **not** to use distributed training is acceptable.
5. **Scaling, cost, and resilience:** Propose a serving architecture, request-scaling strategy, monitoring, approximate costs, and a recovery/rollback plan.

## Today's work plan 20/11/2026

| Activity                    | Expected progress                                                    |
| --------------------------- | -------------------------------------------------------------------- |
| Project inventory           | Identify existing MLOps components, model, and runtime               |
| Resource assessment         | Record dataset/model sizes, CPU/GPU, and training setup              |
| Baseline benchmark          | Run repeated predictions and record latency/throughput               |
| Optimization & architecture | Choose an experiment; sketch scalable serving and training decisions |
| Wrap-up & task split        | Document results, gaps, next steps, and owners                       |

## Step 1 — Inventory your current MLOps system

- Identify the model and framework: PyTorch, TensorFlow, scikit-learn, or other.
- List components already implemented: preprocessing, training, model storage/versioning, prediction API, Docker, CI/CD, and monitoring.
- State where the model runs: local computer, Colab, or GCP.
- Draw a diagram of the **current** system (not the proposed future system).

## Step 2 — Establish resource requirements

- Record dataset size (**approximately 48 MB if applicable to your group**), number of samples, input shape, and model file size.
- Identify CPU/GPU, available RAM/VRAM, batch size, and training duration.
- Explain whether a single machine is sufficient, using evidence from your workload.

## Step 3 — Measure a baseline

- Use a fixed model version, representative test data, and consistent preprocessing.
- Warm up the model; then run repeated predictions.
- Report mean latency in milliseconds and, if possible, p95 latency.
- Report throughput in requests/second, specifying batch size and concurrency.
- Record memory consumption, environment details, and a prediction-quality metric.
- Synchronize GPU operations where necessary to avoid misleading timing results.
- Save the raw measurements and a reproducible benchmark procedure. **Do not invent results.**

### Baseline results template

| Metric                  | Value       | Measurement conditions        |
| ----------------------- | ----------- | ----------------------------- |
| Model and version       | To complete | Framework / artifact version  |
| Hardware                | To complete | CPU / GPU / RAM / VRAM        |
| Model size (MB)         | To measure  | Serialized model file         |
| Mean latency (ms)       | To measure  | Batch size, repetitions       |
| p95 latency (ms)        | To measure  | Batch size, repetitions       |
| Throughput (requests/s) | To measure  | Concurrency and test duration |
| Memory usage (MB)       | To measure  | CPU RAM or GPU VRAM           |
| Classification quality  | To measure  | Metric and held-out dataset   |

## Step 4 — Plan and test one optimization

- Choose a method compatible with **your model and framework**: ONNX Runtime, quantization, or another justified approach.
- Write a hypothesis and define fair comparison conditions **before** testing.
- If feasible, run the optimization and repeat the benchmark using the same test workload and hardware.
- Compare latency, throughput, memory, and classification quality.
- If the optimization is not completed, document prerequisites and blockers. A slower optimized model is a valid result when explained.

| Metric                  | Baseline | Optimized | Interpretation |
| ----------------------- | -------- | --------- | -------------- |
| Mean latency (ms)       | —        | —         | —              |
| p95 latency (ms)        | —        | —         | —              |
| Throughput (requests/s) | —        | —         | —              |
| Memory usage (MB)       | —        | —         | —              |
| Classification quality  | —        | —         | —              |

## Step 5 — Distributed training, scaling, and costs

- Assess whether DDP, FSDP, or ZeRO is justified by training time, model size, dataset size, and communication overhead. **Multi-GPU training is not required.**
- Draw a **proposed** serving architecture including API, model runtime, scaling, monitoring, and failure recovery.
- Estimate CPU/GPU hosting costs using explicit assumptions (region, machine type, usage hours, request volume).
- If using GCP, monitor remaining credits and shut down unnecessary resources. GCP is optional.
- Describe a rollback or recovery procedure. Do **not** claim real autoscaling or deployment unless tested.

## End-of-session checkpoint

Each group should have:

- [ ] Current architecture diagram and resource inventory
- [ ] Baseline results **or** a reproducible benchmark setup with documented blockers
- [ ] One selected optimization and a fair comparison plan
- [ ] Short distributed-training rationale and proposed scaling approach
- [ ] Named responsibilities and next actions

## Suggested repository structure

Keep your existing MLOps repository and add:

```text
ai-infrastructure/
├── README.md
├── architecture/
│   ├── current-system.png
│   └── proposed-system.png
├── benchmarks/
│   ├── benchmark.ipynb
│   └── baseline-results.csv
├── optimization/
│   └── comparison.ipynb
├── distributed-training/
│   └── technical-analysis.md
└── cost-resilience/
    └── infrastructure-plan.md
```

**Note:** These are workshop guidelines for the AI Infrastructure contribution. Any changes to formal assessment requirements remain subject to course coordination.
