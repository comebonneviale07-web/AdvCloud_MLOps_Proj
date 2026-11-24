"""Write a few held-out TEST rides as CSV files to try the web app / API.

    python scripts/export_sample_rides.py --n 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trajclf.config import load_config  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=2, help="rides per class")
args = p.parse_args()
cfg = load_config()
d = cfg["data"]
test_ids = set(np.load(Path(d["processed_dir"]) / "splits.npz")["test_ride_ids"])
labels = pd.read_csv(d["labels_path"]).set_index(d["labels_id_column"])[d["labels_label_column"]]
df = pd.read_parquet(d["raw_path"])
out = ROOT / "data/samples"
out.mkdir(parents=True, exist_ok=True)
for label in cfg["project"]["labels"]:
    ids = [i for i in labels[labels == label].index if i in test_ids][: args.n]
    for rid in ids:
        ride = df[df[d["id_column"]] == rid].sort_values(d["time_column"])[d["feature_columns"]].copy()
        ride["true_label"] = label
        path = out / f"{label.replace(' ', '_')}_{rid[:8]}.csv"
        ride.to_csv(path, index=False)
        print(path)
