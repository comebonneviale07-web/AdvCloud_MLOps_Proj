import numpy as np
import pytest

from trajclf.data import PLUS_STATS, Preprocessor, aggregate_plus_features


def test_aggregate_plus_shape_names_and_no_nan():
    pre = Preprocessor(["ax", "ay", "az", "rx", "ry", "rz"], ["a", "b"], "aggregate_plus")
    pts = np.random.default_rng(0).normal(size=(300, 6))
    pts[5, 2] = np.nan
    x = pre.raw_features(pts)
    assert pre.input_shape == (8 * len(PLUS_STATS),) == x.shape
    assert len(pre.input_names) == x.shape[0] and pre.input_names[-1] == "rot_mag_peak_rate"
    assert np.isfinite(x).all()


def test_aggregate_plus_sees_harsh_driving():
    """A ride with sudden jolts must have a larger jerk than a smooth one."""
    t = np.linspace(0, 10, 500)
    smooth = np.column_stack([np.sin(t)] * 6)
    harsh = smooth.copy()
    harsh[::50] += 5.0
    i = PLUS_STATS.index("jerk_abs_max")
    assert aggregate_plus_features(harsh)[i] > aggregate_plus_features(smooth)[i]


def test_early_stopping_restores_best_epoch():
    torch = pytest.importorskip("torch")
    from trajclf.train import early_stopping_split, train

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 10)).astype(np.float32)
    y = rng.integers(0, 3, size=120).astype(np.int64)  # pure noise: the held-out loss must stop improving
    Xf, yf, Xs, ys = early_stopping_split(X, y, 0.2, 0)
    assert len(ys) == 24 and len(set(map(tuple, Xs)) & set(map(tuple, Xf))) == 0
    cfg = {"project": {"seed": 0, "labels": ["a", "b", "c"]},
           "model": {"type": "mlp", "hidden": 64, "dropout": 0.0},
           "train": {"epochs": 300, "batch_size": 16, "lr": 0.01, "class_weighted_loss": False,
                     "early_stopping_patience": 5, "early_stopping_fraction": 0.2}}
    splits = {"X_train": X, "y_train": y, "X_val": X[:0], "y_val": y[:0]}
    _, stats = train(cfg, splits, torch.device("cpu"), verbose=False)
    assert stats["epochs_run"] < 300 and stats["best_epoch"] <= stats["epochs_run"] - 5
