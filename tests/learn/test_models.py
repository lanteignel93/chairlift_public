"""learn/models.py: ridge in closed form, LightGBM averaged over seeds."""

from __future__ import annotations

import numpy as np
import pytest

from chairlift.learn.models import LightGBM, LightGBMSpec, Ridge, RidgeSpec, Rule, build


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


def test_ridge_intercept_adds_the_training_mean_only_when_asked():
    X, y = data()
    y = y + 3.0
    off, on = Ridge(1e-6).fit(X, y), Ridge(1e-6, intercept=True).fit(X, y)
    np.testing.assert_allclose(on.predict(X) - off.predict(X), y.mean())


def test_lightgbm_deterministic_refits_are_bit_identical_on_many_threads():
    pytest.importorskip("lightgbm")
    X, y = data(3000)
    params = {
        "n_estimators": 30,
        "num_leaves": 8,
        "min_child_samples": 20,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
    }
    s = LightGBMSpec(params=params, seeds=(0, 1), threads=8)
    np.testing.assert_array_equal(build(s).fit(X, y).predict(X), build(s).fit(X, y).predict(X))


def test_rule_finds_the_regime_where_the_position_pays_and_falls_back_to_always_on():
    from chairlift.learn.models import Rule, RuleSpec

    rng = np.random.default_rng(3)
    n = 4000
    regime = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    r = np.where(regime > 0.5, 0.05, -0.02) * 0.1 + 0.01 * rng.standard_normal(n)  # long pays only when regime > 0.5
    m = Rule(RuleSpec(position=1.0)).fit(np.column_stack([noise, regime]), r)
    d = m.describe()
    assert d["input"] == 1 and d["on_when"] == "above" and 0.2 < d["threshold"] < 0.8
    assert set(np.unique(m.predict(np.column_stack([noise, regime])))) == {0.0, 1.0}
    flat = Rule(RuleSpec(position=1.0)).fit(np.column_stack([noise]), 0.001 + 0.01 * rng.standard_normal(n))
    assert flat.describe()["input"] in (-1, 0)  # nothing to time: either always on or a noise split


def test_a_composite_uses_prior_signs_and_learns_the_rest_only_when_asked():
    from chairlift.learn.models import CompositeSpec

    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 3))
    y = 0.1 * X[:, 0] - 0.1 * X[:, 2] + rng.standard_normal(2000)
    m = build(CompositeSpec(signs=(("a", 1.0),))).fit(X, y, ["a", "b", "c"])
    assert m.describe()["weights"] == {"a": 1.0, "b": 0.0, "c": 0.0}
    m = build(CompositeSpec(signs=(("a", 1.0),), learn_signs=True, min_abs_corr=0.05)).fit(X, y, ["a", "b", "c"])
    w = m.describe()["weights"]
    assert w["a"] == 0.5 and w["c"] == -0.5 and w["b"] == 0.0  # b's |corr| is noise, below 0.05: dropped
    flat = build(CompositeSpec()).fit(X, y, ["a", "b", "c"])  # no evidence: no tilt
    assert not np.any(flat.predict(X))


def test_a_ridge_prior_pulls_the_weights_toward_its_direction():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 4))
    y = 0.05 * X[:, 0] + rng.standard_normal(300)  # weak: the free fit is mostly noise
    free = Ridge(1.0).fit(X, y, list("abcd")).w
    pri = Ridge(1.0, prior=(("a", 1.0),)).fit(X, y, list("abcd")).w
    assert abs(pri[0]) / np.abs(pri).sum() > abs(free[0]) / np.abs(free).sum()
    ols = Ridge(1e-12).fit(X, y, list("abcd")).w
    near_ols = Ridge(1e-12, prior=(("a", 1.0),)).fit(X, y, list("abcd")).w
    assert np.allclose(ols, near_ols, atol=1e-8)  # without a penalty the prior has no pull


def test_lightgbm_keeps_a_studys_own_objective():
    pytest.importorskip("lightgbm")
    from chairlift.learn.models import LightGBMSpec

    rng = np.random.default_rng(2)
    X, y = rng.standard_normal((300, 2)), rng.standard_normal(300)
    m = LightGBM(LightGBMSpec(params={"n_estimators": 5, "objective": "huber"})).fit(X, y)
    assert m.models[0].get_params()["objective"] == "huber"
    m = LightGBM(LightGBMSpec(params={"n_estimators": 5})).fit(X, y)
    assert m.models[0].get_params()["objective"] == "regression"


def test_a_rule_keeps_always_in_against_noise_and_takes_a_real_regime():
    from chairlift.learn.models import RuleSpec

    rng = np.random.default_rng(3)
    kept = 0
    for _ in range(40):
        X = rng.standard_normal((750, 5))  # three years of noise states
        r = 0.0005 + 0.01 * rng.standard_normal(750)
        kept += Rule(RuleSpec(position=1.0)).fit(X, r).j == -1
    assert kept >= 36  # without the margin it was 0 of 200 (the review's simulation)
    X = rng.standard_normal((750, 2))
    r = np.where(X[:, 0] > 0, 0.002, -0.002) + 0.01 * rng.standard_normal(750)  # a real regime in input 0
    m = Rule(RuleSpec(position=1.0)).fit(X, r)
    assert m.j == 0 and m.side == 1
    # the old behaviour, margin 0: on noise some rule nearly always beats always-in by luck
    noise_wins = sum(
        Rule(RuleSpec(position=1.0, margin=0.0)).fit(rng.standard_normal((750, 5)), 0.01 * rng.standard_normal(750)).j
        >= 0
        for _ in range(20)
    )
    assert noise_wins >= 18
