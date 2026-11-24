# Distributed training decision: DDP, FSDP or ZeRO?

**Decision: no distributed training.** A single machine (laptop CPU or one Colab T4) is sufficient
for both models. This section justifies the decision with the workload's numbers.

> Legend: **[measured]** = produced by our scripts (fill from `artifacts/bundle/meta.json`,
> `ai-infrastructure/resource-sizing.md`); **[calculated]** = arithmetic from model/dataset sizes;
> **[from data]** = counted in the dataset (docs/FINDINGS.md).

## 1. Workload

| Quantity | MLP (champion candidate) | 1D-CNN (variant) | Type |
|---|---|---|---|
| Training rides | 400 | 400 (or 340 with a validation split) | from data |
| Input per ride | 30 aggregate features | 1024 × 6 resampled readings | config |
| Training tensor size (float32) | 400 × 30 × 4 B = **48 kB** | 400 × 1024 × 6 × 4 B = **9.8 MB** | calculated |
| Raw data | GBG500.parquet 30.2 MB on disk, 1,529,547 readings | same | from data |
| Parameters | **4,229** (30·64+64 + 64·32+32 + 32·5+5) | **94,405** | calculated (checked by `tests/unit/test_model.py`) |
| Training time | *fill: `training_time_s`* | *fill* | measured |
| Peak GPU memory | *fill: `peak_gpu_mem_mb`* | *fill* | measured |

## 2. Memory: does the model fit on one GPU?

Training memory for fp32 + Adam = weights (4 B) + gradients (4 B) + Adam m and v (8 B)
= **16 bytes per parameter**, plus activations (TP3 formula).

| Strategy (2 GPUs) | MLP model states / GPU | CNN model states / GPU |
|---|---|---|
| Single GPU / DDP (replicated) | 4,229 × 16 B = **68 kB** | 94,405 × 16 B = **1.5 MB** |
| ZeRO-1 (shard optimizer states) | 51 kB | 1.1 MB |
| ZeRO-2 (+ shard gradients) | 42 kB | 0.9 MB |
| ZeRO-3 / FSDP (+ shard parameters) | 34 kB | 0.76 MB |

[calculated, idealised — `python ai-infrastructure/sizing.py` reproduces it from the trained bundle]

A T4 has 16 GB of VRAM. The CNN's model states use **~0.01 %** of it; activations for a batch of
32 rides × 1024 steps × 128 channels × 4 B ≤ 17 MB per conv layer are also negligible.
FSDP and ZeRO exist to split model states that do **not** fit on one device (billions of
parameters). Here sharding would save < 1 MB per GPU: **there is no memory problem to solve**.

## 3. Speed: would DDP make training faster?

DDP replicates the model and all-reduces the gradients after every `backward()`:
per step, each GPU sends and receives ≈ 2 × the gradient size (ring all-reduce).

* Per-step compute is tiny: the MLP does ≈8 kFLOPs per ride forward (2 × 4,128 multiply-adds); a whole epoch is
  13 steps of batch 32.
* Fixed per-step costs (kernel launch, NCCL synchronisation, process start-up with `torchrun`)
  are then of the same order as, or larger than, the computation itself. This is the
  **communication-bound case** from the DDP lab (Task 6: `--width 128 --depth 2 --global-batch 32`),
  where 2 GPUs can be **slower** than 1.
* Start-up overhead (spawning processes, NCCL init, data sharding) is paid once per run and is
  likely to exceed the total single-GPU training time (*fill measured value*).
* The bottleneck of the project is **data** (6 Reckless rides), not compute. Faster epochs do not
  improve macro-F1; more labelled rides would.

## 4. When would we revisit the decision?

| Trigger | Response |
|---|---|
| Training data grows by orders of magnitude (e.g. pre-training on SRV3490, 268 MB, 3,490 rides, or continuous data collection) and one epoch takes > ~30 min on one GPU | DDP on one node with several GPUs (throughput scaling, model still replicated) |
| Model grows to hundreds of millions of parameters (e.g. a Transformer on raw 100 Hz sequences) and model states approach GPU memory | FSDP / ZeRO-2/3 |
| Hyper-parameter search or 5-fold CV dominates the time | Run **independent** single-GPU jobs in parallel (embarrassingly parallel); no gradient synchronisation needed |

The third row is the most relevant for us: cross-validation folds (as in
`research/train_neural_network.py`) parallelise without DDP.

## 5. Optional evidence (if time allows)

Reproduce the lab benchmark with the project's model size on one VM with 2 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_benchmark.py --mode baseline --width 64 --depth 2 --global-batch 32 --steps 100
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 train_benchmark.py --mode ddp --width 64 --depth 2 --global-batch 32 --steps 100
```

Report throughput, speedup and parallel efficiency. Only report this if it was actually run.
