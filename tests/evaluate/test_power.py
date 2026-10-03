"""evaluate/power.py: the test window's minimum detectable effect, and the Fama-MacBeth IC t-statistic."""

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest

from chairlift.evaluate.power import fama_macbeth_ic, mde_ic, mde_sharpe, newey_west_var, power_report


def test_mde_matches_the_textbook_numbers():
    # 99% two-sided, 80% power: z = 2.576 + 0.842 = 3.418
    assert mde_sharpe(5.0) == pytest.approx(3.418 / math.sqrt(5), rel=1e-3)
    assert mde_ic(0.15, 60) == pytest.approx(3.418 * 0.15 / math.sqrt(60), rel=1e-3)


def test_newey_west_equals_the_variance_without_lags_and_grows_with_positive_autocorrelation():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    assert newey_west_var(x, 0) == pytest.approx(x.var(), rel=1e-9)
    ar = np.zeros(5000)
    for t in range(1, 5000):
        ar[t] = 0.6 * ar[t - 1] + rng.standard_normal()
    assert newey_west_var(ar, 10) > 2 * ar.var()


def _panel(ic: float, names: int, dates: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(dates):
        x = rng.standard_normal(names)
        y = ic * x + math.sqrt(1 - ic**2) * rng.standard_normal(names)
        d = dt.date(2000, 1, 1) + dt.timedelta(days=30 * t)
        rows.append(pl.DataFrame({"t0": [d] * names, "x": x, "y": y}))
    return pl.concat(rows)


def test_fama_macbeth_finds_a_real_ic_and_not_a_null_one():
    hit = fama_macbeth_ic(_panel(0.1, 300, 120), "x", "y")
    assert hit["mean_ic"] == pytest.approx(0.1, abs=0.02) and hit["t_nw"] > 5
    null = fama_macbeth_ic(_panel(0.0, 300, 120, seed=1), "x", "y")
    assert abs(null["t_nw"]) < 3


def test_power_report_reads_the_ic_volatility_off_the_panel():
    P = _panel(0.02, 400, 60)
    rep = power_report(P, ["x"], "y", "t0", test_dates=60, test_years=5.0)
    # independent names: sd(IC) ≈ 1/√400 = 0.05 and no factor inflation
    assert rep["sd_ic_median"] == pytest.approx(0.05, abs=0.01)
    assert rep["factor_inflation"] == pytest.approx(1.0, abs=0.25)
    assert rep["mde_ic"] == pytest.approx(mde_ic(rep["sd_ic_median"], 60), rel=1e-12)
