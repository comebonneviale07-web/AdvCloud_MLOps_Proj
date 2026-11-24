"""Training with MLflow tracking, MinIO artifact storage and Model Registry.

Usage:
    python -m trajclf.train                 # uses data/processed/splits.npz
    python -m trajclf.train --no-mlflow     # local run, bundle written to artifacts/

What gets logged (MLOps + AI-Infra resource sizing):
  params  : hyper-parameters, dataset sizes, input shape, n_parameters
  metrics : train/val loss & accuracy per epoch, test accuracy / macro-F1,
            random-baseline macro-F1, training time, peak memory
  tags    : hardware (CPU / GPU name, VRAM, RAM), torch version
  artifact: bundle/ (TorchScript + ONNX + preprocess.json + meta.json)
  registry: new version of `trajectory-classifier`; alias `champion` is moved to
            it only if its test macro-F1 beats the current champion.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trajclf.bundle import export_bundle
from trajclf.config import load_config
from trajclf.data import Preprocessor
from trajclf.model import build_model, count_parameters


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, vram_gb=round(props.total_memory / 1e9, 2), cuda=torch.version.cuda)
    return info


def load_splits(cfg: dict) -> tuple[dict, Preprocessor]:
    d = Path(cfg["data"]["processed_dir"])
    npz = np.load(d / "splits.npz")
    return {k: npz[k] for k in npz.files if k.startswith(("X_", "y_"))}, Preprocessor.load(d / "preprocess.json")


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(model(torch.from_numpy(X[i:i + batch]).to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def train(cfg: dict, splits: dict, device: torch.device, log=None) -> tuple[nn.Module, dict]:
    """Plain single-device training loop. ``log(metrics, step)`` is optional."""
    t = cfg["train"]
    model = build_model(cfg, splits["X_train"].shape[1:]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    weight = None
    if t.get("class_weighted_loss"):
        n_classes = len(cfg["project"]["labels"])
        counts = np.bincount(splits["y_train"], minlength=n_classes).astype(np.float32)
        weight = torch.tensor(len(splits["y_train"]) / (n_classes * np.maximum(counts, 1)), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(splits["X_train"]), torch.from_numpy(splits["y_train"])),
        batch_size=t["batch_size"], shuffle=True, num_workers=t.get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    proc = psutil.Process()
    rss_peak = proc.memory_info().rss
    start = time.perf_counter()
    for epoch in range(t["epochs"]):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        rss_peak = max(rss_peak, proc.memory_info().rss)
        metrics = {"train_loss": total / n}
        if len(splits["y_val"]):
            y_val_pred = predict(model, splits["X_val"], device)
            metrics["val_accuracy"] = accuracy_score(splits["y_val"], y_val_pred)
            metrics["val_macro_f1"] = f1_score(splits["y_val"], y_val_pred, average="macro")
        if epoch % 10 == 0 or epoch == t["epochs"] - 1:
            print(f"epoch {epoch + 1:3d}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if log:
            log(metrics, epoch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    stats = {
        "training_time_s": round(elapsed, 2),
        "peak_rss_mb": round(rss_peak / 1e6, 1),
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
        if device.type == "cuda" else 0.0,
    }
    return model, stats


def evaluate(model: nn.Module, splits: dict, labels: list[str], device: torch.device, seed: int) -> dict:
    y_true = splits["y_test"]
    y_pred = predict(model, splits["X_test"], device)
    # "Better than random" check required by the grading grid
    rng = np.random.default_rng(seed)
    y_rand = rng.choice(np.unique(splits["y_train"]), size=len(y_true))
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(labels))), average="macro",
                                        zero_division=0)),
        # Reckless is the class the business cares about (data.txt): track its recall
        "test_reckless_recall": float(
            ((y_pred == labels.index("Reckless")) & (y_true == labels.index("Reckless"))).sum()
            / max((y_true == labels.index("Reckless")).sum(), 1)) if "Reckless" in labels else 0.0,
        "majority_baseline_accuracy": float((y_true == np.bincount(splits["y_train"]).argmax()).mean()),
        "random_baseline_macro_f1": float(f1_score(y_true, y_rand, labels=list(range(len(labels))),
                                                   average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=list(range(len(labels))),
                                        target_names=labels, zero_division=0, output_dict=True),
    }


def promote_if_better(client, name: str, alias: str, version: str, f1: float) -> bool:
    """Move the champion alias to ``version`` if it beats the current champion."""
    try:
        champ = client.get_model_version_by_alias(name, alias)
        champ_f1 = client.get_run(champ.run_id).data.metrics.get("test_macro_f1", -1.0)
    except Exception:  # no champion yet
        champ_f1 = -1.0
    if f1 > champ_f1:
        client.set_registered_model_alias(name, alias, version)
        return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description="Train the trajectory classifier")
    p.add_argument("--config", default=None)
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow, write bundle to --out")
    p.add_argument("--out", default="artifacts/bundle", help="bundle directory for --no-mlflow")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = torch.device(args.device)
    splits, pre = load_splits(cfg)
    labels = cfg["project"]["labels"]

    params = {
        **{f"train.{k}": v for k, v in cfg["train"].items()},
        **{f"model.{k}": v for k, v in cfg["model"].items()},
        "data.mode": pre.mode,
        "data.input_shape": str(tuple(pre.input_shape)),
        "data.n_train": len(splits["y_train"]),
        "data.n_val": len(splits["y_val"]),
        "data.n_test": len(splits["y_test"]),
        "model.n_parameters": count_parameters(build_model(cfg, pre.input_shape)),
        "device": str(device),
    }
    hw = hardware_info()

    if args.no_mlflow:
        model, stats = train(cfg, splits, device)
        ev = evaluate(model, splits, labels, device, cfg["project"]["seed"])
        meta = {"params": params, "hardware": hw, **stats, **{k: v for k, v in ev.items() if k != "report"}}
        out = export_bundle(model, pre, args.out, meta)
        print(json.dumps(meta, indent=2))
        print(f"bundle written to {out}")
        return

    import mlflow
    from mlflow.tracking import MlflowClient

    m = cfg["mlflow"]
    mlflow.set_tracking_uri(m["tracking_uri"])
    mlflow.set_experiment(m["experiment"])
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        raw_stats = Path(cfg["data"]["processed_dir"]) / "raw_stats.json"
        if raw_stats.exists():
            mlflow.log_artifact(str(raw_stats))
            rs = json.loads(raw_stats.read_text())
            mlflow.log_params({"data.n_rides": rs["n_rides"], "data.n_readings": rs["n_readings"],
                               "data.raw_file_mb": rs["raw_file_mb"]})
        mlflow.set_tags({f"hw.{k}": v for k, v in hw.items()})
        model, stats = train(cfg, splits, device, log=lambda mt, step: mlflow.log_metrics(mt, step=step))
        ev = evaluate(model, splits, labels, device, cfg["project"]["seed"])
        mlflow.log_metrics({**stats, **{k: v for k, v in ev.items() if k != "report"}})
        mlflow.log_dict(ev["report"], "classification_report.json")

        meta = {"run_id": run.info.run_id, "params": params, "hardware": hw, **stats,
                **{k: v for k, v in ev.items() if k != "report"}}
        with tempfile.TemporaryDirectory() as tmp:
            bundle = export_bundle(model, pre, Path(tmp) / "bundle", meta)
            for name, size in json.loads((bundle / "meta.json").read_text())["file_size_mb"].items():
                mlflow.log_metric(f"size_mb.{name}", size)
            mlflow.log_artifacts(str(bundle), artifact_path="bundle")

        info = mlflow.pytorch.log_model(model.cpu(), artifact_path="model",
                                        registered_model_name=m["registered_model"])
        client = MlflowClient()
        version = info.registered_model_version or client.get_latest_versions(m["registered_model"])[0].version
        promoted = promote_if_better(client, m["registered_model"], m["champion_alias"],
                                     str(version), ev["test_macro_f1"])
        mlflow.set_tag("promoted_to_champion", promoted)
        print(f"run {run.info.run_id} -> version {version}, champion={promoted}, "
              f"test_macro_f1={ev['test_macro_f1']:.4f} (random {ev['random_baseline_macro_f1']:.4f})")


if __name__ == "__main__":
    main()
