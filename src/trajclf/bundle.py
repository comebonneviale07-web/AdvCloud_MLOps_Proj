"""Model bundle = what the API needs to serve one model version.

    bundle/
    ├── model.ts.pt       TorchScript (no Python class needed at serving time)
    ├── model.onnx        same weights exported to ONNX (optimisation experiment)
    ├── preprocess.json   feature list, seq_len, normalisation stats, labels
    └── meta.json         run id, metrics, parameter count, sizes

The bundle is logged to MLflow (artifacts stored in MinIO / S3) and downloaded
by the API at startup.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from trajclf.data import Preprocessor


def export_bundle(model: torch.nn.Module, pre: Preprocessor, out_dir: str | Path, meta: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    example = torch.zeros(1, *pre.input_shape)
    with torch.no_grad():
        torch.jit.trace(model, example).save(str(out / "model.ts.pt"))
        kwargs = dict(input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
        try:  # recent torch defaults to the dynamo exporter; the TorchScript exporter is simpler here
            torch.onnx.export(model, example, str(out / "model.onnx"), dynamo=False, **kwargs)
        except TypeError:  # older torch without the `dynamo` argument
            torch.onnx.export(model, example, str(out / "model.onnx"), **kwargs)
    pre.save(out / "preprocess.json")
    meta = dict(meta)
    meta["file_size_mb"] = {
        f.name: round(f.stat().st_size / 1e6, 4) for f in out.iterdir() if f.suffix in {".pt", ".onnx"}
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out
