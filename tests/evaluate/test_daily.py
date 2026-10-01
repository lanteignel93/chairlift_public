"""evaluate/daily.py: the overlapping book's daily P&L and its block-bootstrap statistics."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from chairlift.evaluate.daily import DailyStatsSpec, book_stats, daily_book, daily_stats, ic_by_year


def test_daily_pnl_is_the_sum_of_path_increments_long_minus_short():
    d = [dt.date(2020, 1, k) for k in (1, 2, 3)]
    positions = pl.DataFrame({"root": ["A", "B"], "t0": [d[0], d[0]], "side": ["long", "short"]})
    paths = pl.DataFrame(
        {"root": ["A"] * 3 + ["B"] * 3, "t0": [d[0]] * 6, "date": d * 2, "pnl": [0.0, 1.0, 3.0, 0.0, -1.0, -1.0]}
    )
    day = daily_book(positions, paths)
    assert day["long"].to_list() == [0.0, 1.0, 2.0] and day["short"].to_list() == [0.0, -1.0, 0.0]
    assert day["ls"].to_list() == [0.0, 2.0, 2.0] and day["n_long"].to_list() == [1, 1, 1]
    assert daily_book(positions, paths, end=d[1]).height == 2


def test_stats_sharpe_ci_brackets_the_point_and_is_seeded():
    rng = np.random.default_rng(0)
    dates = [dt.date(2018, 1, 1) + dt.timedelta(days=i) for i in range(1000)]
    x = 0.1 + rng.standard_normal(1000)
    s = DailyStatsSpec(n_boot=500, seed=3)
    a, b = daily_stats(x, dates, s), daily_stats(x, dates, s)
    assert a == b
    assert a["sharpe_ci"][0] < a["sharpe"] < a["sharpe_ci"][1]
    assert abs(a["sharpe"] - 0.1 * np.sqrt(252)) < 1.0 and a["n_years"] == 3


def test_book_stats_and_ic_by_year():
    dates = [dt.date(2018, 1, 1) + dt.timedelta(days=i) for i in range(300)]
    day = pl.DataFrame(
        {
            "date": dates,
            "long": np.ones(300),
            "short": np.zeros(300),
            "ls": np.linspace(-1, 1, 300),
            "n_long": 3,
            "n_short": 3,
        }
    )
    out = book_stats(day, DailyStatsSpec(n_boot=100))
    assert out["avg_live_long"] == 3.0 and set(out) >= {"long", "short", "ls"}
    f = pl.DataFrame(
        {
            "t0": [1, 1, 1, 2, 2, 2],
            "y_year": [2018] * 3 + [2019] * 3,
            "pred": [1, 2, 3, 1, 2, 3],
            "y_rank": [0.0, 0.5, 1.0, 1.0, 0.5, 0.0],
        }
    )
    ic = ic_by_year(f)
    assert ic["oos_ic_by_year"] == {2018: 1.0, 2019: -1.0} and ic["oos_ic_worst_year"] == -1.0


def test_time_series_ic_reads_across_rows():
    from chairlift.evaluate.daily import ts_ic_by_year

    d = [dt.date(2018, 1, 1) + dt.timedelta(days=i) for i in range(4)] + [
        dt.date(2019, 1, 1) + dt.timedelta(days=i) for i in range(4)
    ]
    f = pl.DataFrame({"date": d, "pred": [1.0, 2.0, 3.0, 4.0] * 2, "y": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]})
    ic = ts_ic_by_year(f)
    assert ic["oos_ic_by_year"] == {2018: 1.0, 2019: -1.0} and ic["oos_ic_all"] == pytest.approx(0.0)
