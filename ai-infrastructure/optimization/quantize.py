"""INT8 quantization of the ONNX model with ONNX Runtime.

    python ai-infrastructure/optimization/quantize.py --bundle artifacts/bundle [--mode static]

static  (default): QDQ static quantization calibrated on 200 TRAINING rides
                   (never the test split). Best suited to a CNN (Conv weights
                   and activations in INT8).
dynamic          : weights-only INT8, activations quantized on the fly.
Writes bundle/model.int8.onnx, then run benchmark.py with --runtimes ... onnx-int8.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class TrainReader:
    """Feeds calibration batches to ONNX Runtime."""

    def __init__(self, X: np.ndarray, n: int = 200, bs: int = 8):
        self.batches = iter([{"input": X[i:i + bs]} for i in range(0, min(n, len(X)), bs)])

    def get_next(self):
        return next(self.batches, None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default=str(ROOT / "artifacts/bundle"))
    p.add_argument("--splits", default=str(ROOT / "data/processed/splits.npz"))
    p.add_argument("--mode", choices=["static", "dynamic"], default="static")
    args = p.parse_args()

    from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic, quantize_static

    src = Path(args.bundle) / "model.onnx"
    pre = Path(args.bundle) / "model.pre.onnx"
    dst = Path(args.bundle) / "model.int8.onnx"
    quant_pre_process(str(src), str(pre))  # shape inference + graph cleanup
    if args.mode == "dynamic":
        quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QInt8)
    else:
        X = np.load(args.splits)["X_train"].astype(np.float32)
        quantize_static(str(pre), str(dst), TrainReader(X), activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8)
    pre.unlink(missing_ok=True)
    before, after = src.stat().st_size / 1e6, dst.stat().st_size / 1e6
    print(f"{src.name}: {before:.3f} MB -> {dst.name}: {after:.3f} MB ({args.mode})")


if __name__ == "__main__":
    main()
