"""HTTP load test of the deployed API (end-to-end latency incl. network + JSON + preprocessing).

    python ai-infrastructure/benchmarks/load_test.py --url http://localhost:8000 \
        --concurrency 1 4 16 --duration 30

Stdlib only (threads + urllib) so it runs anywhere. Uses real rides
from data/raw/GBG500.parquet if present, else random rides of realistic size.
Writes results/load-test.csv: concurrency, requests, errors, req/s, mean/p50/p95/p99 ms.
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results"


def make_payloads(url: str, n: int = 50, ride_len: int = 3000) -> list[bytes]:
    """Rides of realistic size (GBG500 median ≈ 3,000 readings × 6 sensors ≈ 100 kB of JSON).

    Uses real test rides from data/raw if available, else random readings of the same size.
    """
    info = json.loads(urllib.request.urlopen(f"{url}/model", timeout=10).read())
    raw = Path(__file__).resolve().parents[2] / "data/raw/GBG500.parquet"
    if raw.exists():
        import pandas as pd
        df = pd.read_parquet(raw).sort_values(["ride_id", "time_index"])
        rides = [g[info["features"]].fillna(0).round(5).to_numpy().tolist()
                 for _, g in list(df.groupby("ride_id"))[:n]]
    else:
        rng = np.random.default_rng(0)
        rides = [rng.normal(size=(ride_len, len(info["features"]))).round(5).tolist() for _ in range(n)]
    return [json.dumps({"points": r}).encode() for r in rides]


def run_level(url: str, payloads: list[bytes], concurrency: int, duration: float) -> dict:
    lat, errors, lock = [], [0], threading.Lock()
    stop = time.perf_counter() + duration

    def user(i: int) -> None:
        k = i
        while time.perf_counter() < stop:
            req = urllib.request.Request(f"{url}/predict", payloads[k % len(payloads)],
                                         {"Content-Type": "application/json"})
            s = time.perf_counter()
            try:
                urllib.request.urlopen(req, timeout=30).read()
                with lock:
                    lat.append((time.perf_counter() - s) * 1000)
            except Exception:
                with lock:
                    errors[0] += 1
            k += concurrency

    threads = [threading.Thread(target=user, args=(i,)) for i in range(concurrency)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    a = np.asarray(lat) if lat else np.asarray([np.nan])
    return {"concurrency": concurrency, "duration_s": round(elapsed, 2), "requests": len(lat),
            "errors": errors[0], "throughput_req_s": len(lat) / elapsed, "mean_ms": float(np.mean(a)),
            "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 16])
    p.add_argument("--duration", type=float, default=30)
    p.add_argument("--tag", default="", help="label stored in the CSV, e.g. 'torch-2pods'")
    args = p.parse_args()
    payloads = make_payloads(args.url)
    run_level(args.url, payloads, 2, 3)  # warm-up
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in args.concurrency:
        r = {"tag": args.tag, **run_level(args.url, payloads, c, args.duration)}
        print(json.dumps(r))
        rows.append(r)
    path = OUT / "load-test.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"appended to {path}")


if __name__ == "__main__":
    main()
