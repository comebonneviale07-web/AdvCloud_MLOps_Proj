import numpy as np
import pandas as pd
import pytest

from trajclf.data import STATS, Preprocessor, aggregate_features


def make_pre(mode="sequence", seq_len=8, n=3):
    return Preprocessor([f"f{i}" for i in range(n)], ["a", "b"], mode, seq_len, [0.0] * n, [1.0] * n)


def test_resample_shape_and_endpoints():
    out = make_pre(seq_len=10).resample(np.stack([np.linspace(0, 1, 37)] * 3, axis=1))
    assert out.shape == (10, 3)
    assert np.allclose(out[0], 0) and np.allclose(out[-1], 1)


def test_resample_interpolates_nan():
    pts = np.stack([np.arange(5.0)] * 3, axis=1)
    pts[2, 0] = np.nan
    out = make_pre(seq_len=5).resample(pts)
    assert not np.isnan(out).any() and out[2, 0] == pytest.approx(2.0)


def test_wrong_feature_count_raises():
    with pytest.raises(ValueError):
        make_pre().resample(np.ones((4, 2)))


def test_aggregate_matches_research_prepare_features():
    """Same 30 features as research/prepare_features.py (pandas agg, NaN-ignoring)."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 6))
    pts[[3, 50], [0, 4]] = np.nan
    cols = ["ax", "ay", "az", "rx", "ry", "rz"]
    expected = pd.DataFrame(pts, columns=cols).agg(list(STATS))  # rows = stats, columns = sensors
    expected = np.asarray([expected.loc[s, c] for c in cols for s in STATS])
    assert np.allclose(aggregate_features(pts.astype(np.float32)), expected, atol=1e-5)


def test_aggregate_input_shape_and_names():
    pre = make_pre(mode="aggregate", n=6)
    assert pre.input_shape == (30,) and pre.input_names[:2] == ["f0_mean", "f0_std"]


def test_save_load_roundtrip(tmp_path):
    pre = make_pre()
    pre.save(tmp_path / "p.json")
    assert Preprocessor.load(tmp_path / "p.json") == pre


def test_build_datasets_stratified_and_shapes(cfg, datasets):
    n = sum(len(datasets[k]) for k in ("y_train", "y_val", "y_test"))
    assert n == datasets["raw_stats"]["n_rides"]
    assert len(np.unique(datasets["y_test"])) == len(cfg["project"]["labels"])
    assert datasets["X_train"].shape[1:] == datasets["preprocessor"].input_shape
    assert not np.isnan(datasets["X_train"]).any()


def test_normalisation_fitted_on_train_only(datasets):
    x = datasets["X_train"].reshape(-1, datasets["X_train"].shape[-1])
    assert np.allclose(x.mean(0), 0, atol=1e-3)
