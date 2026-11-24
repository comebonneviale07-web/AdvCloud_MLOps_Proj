"""Hyper-parameter sweep: every variant becomes an MLflow run, compared on 5-fold CV macro-F1.

    python scripts/sweep.py                 # MLP variants (configs/config.yaml)
    python scripts/sweep.py --dry-run       # print the commands only

Selection uses cross-validation on the 400 training rides; the 100 test rides are only reported,
never used to choose (see docs/FINDINGS.md). The champion alias moves only if CV improves.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# name -> config overrides (everything else = configs/config.yaml)
VARIANTS: dict[str, list[str]] = {
    "baseline": [],
    "epochs-30": ["train.epochs=30"],
    "epochs-50": ["train.epochs=50"],
    "epochs-200": ["train.epochs=200"],
    "dropout-0.5": ["model.dropout=0.5"],
    "hidden-32": ["model.hidden=32"],
    "hidden-128": ["model.hidden=128"],
    "lr-3e-4": ["train.lr=0.0003"],
    "weight-decay-1e-3": ["train.weight_decay=0.001"],
    "regularised": ["model.dropout=0.5", "train.weight_decay=0.001", "train.epochs=50"],
    "no-class-weights": ["train.class_weighted_loss=false"],
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", help="run only these variant names")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    # PYTHONUTF8: on Windows, captured output defaults to cp1252 and MLflow's "🏃 View run" line crashes
    env = {**os.environ, "PYTHONPATH": "src", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    for name, sets in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        cmd = [sys.executable, "-m", "trajclf.train", "--run-name", f"sweep-{name}"]
        for item in sets:
            cmd += ["--set", item]
        print(">>", " ".join(cmd), flush=True)
        if not args.dry_run:
            out = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            lines = [ln for ln in out.stdout.splitlines() if ln.startswith(("CV macro-F1", "run "))]
            print("\n".join(lines) if out.returncode == 0 else out.stderr[-2000:], flush=True)
    if args.dry_run:
        return

    import mlflow

    runs = mlflow.search_runs(experiment_names=["trajectory-classification"],
                              filter_string="tags.mlflow.runName LIKE 'sweep-%' and attributes.status = 'FINISHED'",
                              order_by=["metrics.cv_macro_f1_mean DESC"])
    cols = ["tags.mlflow.runName", "metrics.cv_macro_f1_mean", "metrics.cv_macro_f1_std",
            "metrics.train_macro_f1", "metrics.overfit_gap_f1", "metrics.test_macro_f1"]
    print("\nRanked by CV macro-F1 (selection metric):")
    print(runs[[c for c in cols if c in runs]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
