"""evaluate/deflated.py: the best of many trials is charged for the number tried."""

from __future__ import annotations

import pytest

from chairlift.evaluate.deflated import deflated_sharpe, expected_max_sharpe, probabilistic_sharpe


def test_one_trial_is_the_probabilistic_sharpe_against_zero():
    d = deflated_sharpe([1.0], n_obs=1000)
    assert d["expected_max_sharpe_null"] == 0.0 and d["dsr"] == pytest.approx(d["psr_vs_zero"])
    # annual Sharpe 1 is 1/√252 per day; over 1,000 days z = (1/√252)·√999 ≈ 1.99, Φ(z) ≈ 0.977
    assert d["dsr"] == pytest.approx(0.9766, abs=5e-4)


def test_more_trials_raise_the_bar_and_lower_the_dsr():
    few = deflated_sharpe([1.0, 0.2, 0.4], n_obs=1000)
    many = deflated_sharpe([1.0] + [0.2, 0.4, -0.1, 0.6, 0.0, 0.3, -0.3, 0.5, 0.1] * 5, n_obs=1000)
    assert many["expected_max_sharpe_null"] > few["expected_max_sharpe_null"] > 0
    assert many["dsr"] < few["dsr"]


def test_expected_max_grows_with_trials_and_spread():
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01) > 0
    assert expected_max_sharpe(10, 0.04) == pytest.approx(2 * expected_max_sharpe(10, 0.01))
    assert probabilistic_sharpe(0.0, 500) == pytest.approx(0.5)
