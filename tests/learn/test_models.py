"""learn/models.py: ridge in closed form, LightGBM averaged over seeds."""

from __future__ import annotations

import numpy as np
import pytest

from chairlift.learn.models import LightGBMSpec, Ridge, RidgeSpec, build


def data(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    return X, X @ np.array([1.0, -0.5, 0.0]) + 0.1 * rng.standard_normal(n)


def test_ridge_recovers_coefficients_on_the_standardised_scale():
    X, y = data()
    m = build(RidgeSpec(alpha=1e-6)).fit(X, y)
    assert isinstance(m, Ridge)
    np.testing.assert_allclose(m.w / X.std(0), [1.0, -0.5, 0.0], atol=0.02)
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.99


def test_ridge_penalty_scales_with_n_so_alpha_means_the_same_at_any_size():
    X, y = data(4000)
    small = Ridge(0.5).fit(X[:2000], y[:2000]).w
    big = Ridge(0.5).fit(X, y).w
    np.testing.assert_allclose(small, big, atol=0.02)
    assert np.abs(Ridge(10.0).fit(X, y).w).sum() < np.abs(big).sum()


def test_lightgbm_is_seeded_and_averages_seeds():
    pytest.importorskip("lightgbm")
    X, y = data(800)
    params = {"n_estimators": 20, "num_leaves": 4, "min_child_samples": 20, "bagging_fraction": 0.8, "bagging_freq": 1}
    a = build(LightGBMSpec(params=params, seeds=(0, 1), threads=1)).fit(X, y)
    b = build(LightGBMSpec(params=params, seeds=(0, 1), threads=1)).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert len(a.describe()["gain_importance_mean"]) == 3
    assert np.corrcoef(a.predict(X), y)[0, 1] > 0.8
