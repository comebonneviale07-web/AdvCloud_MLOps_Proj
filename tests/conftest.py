"""Shared fixtures: tiny synthetic datasets + trained bundles (no MLflow server, no real data)."""
import copy

import pytest

from trajclf.config import load_config
from trajclf.data import build_datasets, save_processed, write_synthetic


def _cfg(tmp, mode, model_type):
    c = copy.deepcopy(load_config())
    c["data"].update(raw_path=str(tmp / "raw.csv"), labels_path=str(tmp / "labels.csv"),
                     processed_dir=str(tmp / "processed"), mode=mode, seq_len=64)
    c["model"]["type"] = model_type
    c["train"].update(epochs=3, batch_size=16, num_workers=0)
    write_synthetic(c, n_per_class=12)
    return c


@pytest.fixture(scope="session", params=[("aggregate", "mlp"), ("sequence", "cnn")], ids=["mlp", "cnn"])
def cfg(request, tmp_path_factory):
    return _cfg(tmp_path_factory.mktemp(request.param[1]), *request.param)


@pytest.fixture(scope="session")
def datasets(cfg):
    ds = build_datasets(cfg)
    save_processed(ds, cfg["data"]["processed_dir"])
    return ds


@pytest.fixture(scope="session")
def bundle(cfg, datasets, tmp_path_factory):
    torch = pytest.importorskip("torch")
    from trajclf.bundle import export_bundle
    from trajclf.train import evaluate, train

    splits = {k: v for k, v in datasets.items() if k.startswith(("X_", "y_"))}
    model, stats = train(cfg, splits, torch.device("cpu"))
    ev = evaluate(model, splits, cfg["project"]["labels"], torch.device("cpu"), 0)
    return export_bundle(model, datasets["preprocessor"], tmp_path_factory.mktemp("bundle"),
                         {**stats, "test_macro_f1": ev["test_macro_f1"]})
