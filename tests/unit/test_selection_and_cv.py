"""Regression tests for the two leakage fixes (CV standardisation, champion selection)."""
import numpy as np
import pytest

pytest.importorskip("torch")

from trajclf.train import fold_standardise, selection_metric  # noqa: E402


def test_fold_standardise_ignores_held_out_rides():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(20, 3)).astype(np.float32)
    fit_idx, val_idx = np.arange(15), np.arange(15, 20)
    X1 = fold_standardise(raw, fit_idx)
    raw2 = raw.copy()
    raw2[val_idx] *= 1000          # change ONLY the held-out rides
    X2 = fold_standardise(raw2, fit_idx)
    assert np.allclose(X1[fit_idx], X2[fit_idx])          # scaling unaffected by held-out data
    assert np.allclose(X1[fit_idx].mean(0), 0, atol=1e-5)  # fitted on the fitting rides


def test_selection_never_uses_test_score():
    assert selection_metric({"test_macro_f1": 0.99}) is None
    assert selection_metric({"cv_macro_f1_mean": 0.3, "test_macro_f1": 0.99}) == 0.3
