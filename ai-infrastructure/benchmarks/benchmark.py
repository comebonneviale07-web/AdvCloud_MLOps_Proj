"""Model-level inference benchmark (step 3 baseline + step 4 comparison).

Each runtime is measured in its OWN subprocess so that memory numbers are not
polluted by the other runtime. Same bundle, same test split, same batch sizes,
same thread count for every runtime = fair comparison.

    python ai-infrastructure/benchmarks/benchmark.py --bundle artifacts/bundle \
        --runtimes torch onnx onnx-int8 --batch-sizes 1 8 32 --threads 1

Outputs (in ai-infrastructure/benchmarks/results/):
    raw-latencies.csv     every timed call (reproducibility, re-analysis)
    baseline-results.csv  one row per (runtime, batch size): mean/p50/p95/p99, throughput, memory, quality
    preprocess-results.csv  time to turn ONE raw ride (≈3,000 readings) into model input
    environment.json      hardware + software versions
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(__file__).resolve().parent / "results"


def environment() -> dict:
    env = {"platform": platform.platform(), "python": platform.python_version(),
           "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
           "ram_gb": round(psutil.virtual_memory().total / 1e9, 2)}
    for mod in ("torch", "onnxruntime", "numpy"):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except ImportError:
        pass
    return env


def worker(args) -> None:
    """Runs in a subprocess: benchmark ONE runtime, print JSON on stdout."""
    from sklearn.metrics import accuracy_score, f1_score

    from trajclf.predictor import Predictor

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    if args.device == "cuda":
        import torch
        model = torch.jit.load(str(Path(args.bundle) / "model.ts.pt"), map_location="cuda").eval()

        def forward(x):
            with torch.inference_mode():
                out = model(torch.from_numpy(x).cuda())
                torch.cuda.synchronize()  # GPU calls are async: sync before stopping the clock
                return out.cpu().numpy()
        torch.cuda.reset_peak_memory_stats()
    else:
        pred = Predictor(args.bundle, args.runtime, num_threads=args.threads)
        forward = pred.logits
    load_s = time.perf_counter() - t0

    splits = np.load(args.splits)
    X, y = splits["X_test"].astype(np.float32), splits["y_test"]

    # Prediction quality on the held-out test split
    y_hat = np.concatenate([forward(X[i:i + 64]).argmax(1) for i in range(0, len(X), 64)])
    quality = {"test_accuracy": float(accuracy_score(y, y_hat)),
               "test_macro_f1": float(f1_score(y, y_hat, average="macro"))}

    rows, raw = [], []
    rng = np.random.default_rng(0)
    for bs in args.batch_sizes:
        idx = rng.integers(0, len(X), size=bs)
        xb = np.ascontiguousarray(X[idx])
        for _ in range(args.warmup):
            forward(xb)
        lat = np.empty(args.reps)
        for r in range(args.reps):
            s = time.perf_counter()
            forward(xb)
            lat[r] = (time.perf_counter() - s) * 1000
        raw += [{"runtime": args.runtime, "device": args.device, "batch_size": bs, "rep": r, "latency_ms": v}
                for r, v in enumerate(lat)]
        rows.append({
            "runtime": args.runtime, "device": args.device, "threads": args.threads, "batch_size": bs,
            "repetitions": args.reps, "warmup": args.warmup,
            "mean_ms": lat.mean(), "p50_ms": np.percentile(lat, 50), "p95_ms": np.percentile(lat, 95),
            "p99_ms": np.percentile(lat, 99), "std_ms": lat.std(),
            "throughput_samples_s": bs * 1000 / lat.mean(),
        })
    mem = {"rss_after_load_mb": (proc.memory_info().rss) / 1e6,
           "rss_model_delta_mb": (proc.memory_info().rss - rss_before) / 1e6, "load_time_s": load_s}
    if args.device == "cuda":
        import torch
        mem["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1e6
    size_file = {"torch": "model.ts.pt", "onnx": "model.onnx", "onnx-int8": "model.int8.onnx"}[args.runtime]
    mem["model_file_mb"] = (Path(args.bundle) / size_file).stat().st_size / 1e6
    for r in rows:
        r.update(mem)
        r.update(quality)
    print(json.dumps({"rows": rows, "raw": raw}))


def preprocess_benchmark(bundle: str, reps: int) -> list[dict]:
    """Raw ride -> model input, on real test rides (needs data/raw + config)."""
    from trajclf.config import load_config
    from trajclf.data import Preprocessor, load_raw

    cfg = load_config()
    if not Path(cfg["data"]["raw_path"]).exists():
        return []
    pre = Preprocessor.load(Path(bundle) / "preprocess.json")
    ids, rides, _ = load_raw(cfg)
    test_ids = set(np.load(Path(cfg["data"]["processed_dir"]) / "splits.npz")["test_ride_ids"])
    rides = [r for i, r in zip(ids, rides, strict=True) if i in test_ids]
    lat = []
    for k in range(reps):
        r = rides[k % len(rides)]
        s = time.perf_counter()
        pre.transform(r)
        lat.append((time.perf_counter() - s) * 1000)
    lat = np.asarray(lat[min(20, reps // 10):])  # drop warm-up calls
    return [{"mode": pre.mode, "rides": len(rides), "calls": len(lat),
             "mean_readings_per_ride": float(np.mean([len(r) for r in rides])),
             "mean_ms": lat.mean(), "p50_ms": np.percentile(lat, 50), "p95_ms": np.percentile(lat, 95)}]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default=str(ROOT / "artifacts/bundle"))
    p.add_argument("--splits", default=str(ROOT / "data/processed/splits.npz"))
    p.add_argument("--runtimes", nargs="+", default=["torch", "onnx"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cuda only for runtime torch")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--reps", type=int, default=300)
    p.add_argument("--runtime", help=argparse.SUPPRESS)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    if args.worker:
        worker(args)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows, all_raw = [], []
    for rt in args.runtimes:
        cmd = [sys.executable, __file__, "--worker", "--runtime", rt, "--bundle", args.bundle,
               "--splits", args.splits, "--device", args.device, "--threads", str(args.threads),
               "--warmup", str(args.warmup), "--reps", str(args.reps), "--batch-sizes",
               *map(str, args.batch_sizes)]
        print(f"benchmarking {rt} on {args.device} ...", flush=True)
        res = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.splitlines()[-1])
        all_rows += res["rows"]
        all_raw += res["raw"]

    for name, data in (("baseline-results.csv", all_rows), ("raw-latencies.csv", all_raw)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    pre_rows = preprocess_benchmark(args.bundle, args.reps)
    if pre_rows:
        with open(out / "preprocess-results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pre_rows[0]))
            w.writeheader()
            w.writerows(pre_rows)
        print(f"preprocessing one raw ride: mean {pre_rows[0]['mean_ms']:.3f} ms, p95 {pre_rows[0]['p95_ms']:.3f} ms")
    env = environment()
    env["command"] = " ".join(sys.argv)
    (out / "environment.json").write_text(json.dumps(env, indent=2))

    print(f"\n{'runtime':10} {'bs':>3} {'mean ms':>8} {'p95 ms':>8} {'samples/s':>10} {'RSS MB':>7} {'F1':>6}")
    for r in all_rows:
        print(f"{r['runtime']:10} {r['batch_size']:>3} {r['mean_ms']:8.3f} {r['p95_ms']:8.3f} "
              f"{r['throughput_samples_s']:10.1f} {r['rss_after_load_mb']:7.1f} {r['test_macro_f1']:6.3f}")
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
