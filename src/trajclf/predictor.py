"""Model loading + inference, with two interchangeable runtimes.

MODEL_RUNTIME=torch  -> TorchScript (baseline)
MODEL_RUNTIME=onnx   -> ONNX Runtime (optimisation experiment)

The model comes either from a local bundle directory (MODEL_DIR) or from the
MLflow Model Registry (MODEL_NAME + MODEL_ALIAS, artifacts in MinIO).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from trajclf.data import Preprocessor


class Predictor:
    def __init__(self, bundle_dir: str | Path, runtime: str = "torch", num_threads: int | None = None):
        self.bundle_dir = Path(bundle_dir)
        self.runtime = runtime
        self.pre = Preprocessor.load(self.bundle_dir / "preprocess.json")
        meta_path = self.bundle_dir / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if runtime == "torch":
            import torch

            if num_threads:
                torch.set_num_threads(num_threads)
            self._torch = torch
            self._model = torch.jit.load(str(self.bundle_dir / "model.ts.pt"), map_location="cpu").eval()
        elif runtime in {"onnx", "onnx-int8"}:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            if num_threads:
                opts.intra_op_num_threads = num_threads
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            fname = "model.int8.onnx" if runtime == "onnx-int8" else "model.onnx"
            self._sess = ort.InferenceSession(str(self.bundle_dir / fname), opts,
                                              providers=["CPUExecutionProvider"])
        else:
            raise ValueError(f"unknown runtime {runtime!r}")

    @property
    def labels(self) -> list[str]:
        return self.pre.labels

    def logits(self, x: np.ndarray) -> np.ndarray:
        """x: preprocessed (B, *pre.input_shape) float32 -> logits (B, C)."""
        if self.runtime == "torch":
            with self._torch.inference_mode():
                return self._model(self._torch.from_numpy(x)).numpy()
        return self._sess.run(None, {"input": x})[0]

    def predict(self, rides: list[np.ndarray]) -> list[dict]:
        return self.postprocess(self.logits(self.pre.transform_batch(rides)))

    def postprocess(self, z: np.ndarray) -> list[dict]:
        z = z - z.max(axis=1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return [
            {"label": self.labels[int(p.argmax())], "confidence": float(p.max()),
             "probabilities": {lab: float(v) for lab, v in zip(self.labels, p, strict=True)}}
            for p in probs
        ]


def download_from_registry(name: str, alias: str, dst: str | Path) -> tuple[Path, str]:
    """Download the bundle of ``models:/name@alias`` from MLflow/MinIO."""
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="bundle", dst_path=str(dst))
    return Path(path), str(mv.version)


def load_predictor_from_env() -> tuple[Predictor, str]:
    runtime = os.environ.get("MODEL_RUNTIME", "torch")
    threads = int(os.environ["NUM_THREADS"]) if os.environ.get("NUM_THREADS") else None
    if local := os.environ.get("MODEL_DIR"):
        return Predictor(local, runtime, threads), "local"
    name = os.environ.get("MODEL_NAME", "trajectory-classifier")
    alias = os.environ.get("MODEL_ALIAS", "champion")
    path, version = download_from_registry(name, alias, os.environ.get("MODEL_CACHE", "/tmp/model"))
    return Predictor(path, runtime, threads), version
