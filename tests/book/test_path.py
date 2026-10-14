"""book/path.py: weights from earlier predictions only, path P&L summed by date, books marked on the calendar."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from chairlift.book.path import PathSizing, daily_pnl, final_pnl, on_calendar, sizes

D = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(10)]


def _preds() -> pl.DataFrame:
    """Fold 2020 trained on three rows (predictions 1, 2, 3), then tests four entries; fold 2021 trains on two rows
    and tests two entries."""
    rows = [(D[0], 1.0, 2020, True), (D[1], 2.0, 2020, True), (D[2], 3.0, 2020, True)]
    rows += [(D[3], 2.5, 2020, False), (D[4], 0.5, 2020, False), (D[5], 4.0, 2020, False), (D[6], 2.5, 2020, False)]
    rows += [(D[0], 10.0, 2021, True), (D[1], 20.0, 2021, True), (D[7], 15.0, 2021, False), (D[8], 0.0, 2021, False)]
    return pl.DataFrame(rows, schema=["t0", "pred", "fold", "is_train"], orient="row")


def test_rank_weights_count_only_earlier_predictions():
    w = sizes(_preds(), PathSizing(rule="rank"))
    assert w["t0"].to_list() == [D[3], D[4], D[5], D[6], D[7], D[8]]
    # 2.5 among {1, 2, 3}: two below; 0.5 among {1, 2, 3, 2.5}: none; 4.0 among five: all five; 2.5 among six: three
    # (strictly below); fold 2021: 15 among {10, 20} and the four earlier test predictions: 2.5, 0.5, 4, 2.5, 10 below
    assert np.allclose(w["w"].to_numpy(), [2 * 2 / 3, 0.0, 2 * 5 / 5, 2 * 3 / 6, 2 * 5 / 6, 0.0])


def test_the_other_rules():
    p = _preds()
    assert sizes(p, PathSizing(rule="constant"))["w"].to_list() == [1.0] * 6
    terc = sizes(p, PathSizing(rule="tercile"))["w"].to_list()
    assert terc == [2.0, 0.0, 2.0, 1.0, 2.0, 0.0]  # a percentile of exactly 2/3 is the top third
    inv = sizes(p, PathSizing(rule="inverse"))["w"].to_numpy()
    assert np.isclose(inv[0], 2.0 / 2.5) and inv[5] == 1.0  # median of {1, 2, 3} over 2.5; a zero prediction keeps 1
    direct = sizes(p, PathSizing(rule="direct"))["w"].to_numpy()
    assert np.isclose(direct[0], 2.5 / 2.0) and np.isclose(direct[1], 0.5 / (8.5 / 4))  # this fold's history only


def _trades() -> pl.DataFrame:
    """Two overlapping trades: A entered D0, marked -1 (entry cost), then +2, then +5; B entered D1, marked 0, -3."""
    return pl.DataFrame(
        {
            "t0": [D[0], D[0], D[0], D[1], D[1]],
            "date": [D[0], D[1], D[2], D[1], D[2]],
            "pnl": [-1.0, 2.0, 5.0, 0.0, -3.0],
            "size": [2.0, 2.0, 2.0, 1.0, 1.0],
        }
    )


def test_daily_pnl_sums_each_live_trade_s_change_and_charges_the_entry_mark_on_its_day():
    s = PathSizing(unit="size", scale=10.0)
    w = pl.DataFrame({"t0": [D[0], D[1]], "w": [0.5, 2.0]})
    d = daily_pnl(_trades(), w, s)
    # D0: A's first mark -1 → -1 × 2 × 0.5 × 10; D1: A +3 → 30, B 0; D2: A +3 → 30, B -3 × 1 × 2 × 10 = -60
    assert d.to_dicts() == [{"date": D[0], "pnl": -10.0}, {"date": D[1], "pnl": 30.0}, {"date": D[2], "pnl": -30.0}]
    only_b = daily_pnl(_trades(), w.filter(pl.col("t0") == D[1]), s)
    assert only_b["date"].to_list() == [D[1], D[2]]  # a trade without a weight is not in the book


def test_books_are_marked_on_every_calendar_day_and_a_missing_pnl_day_is_refused():
    cal = pl.DataFrame({"date": D[:6]})
    d = pl.DataFrame({"date": [D[1], D[3]], "pnl": [1.0, 2.0]})
    out = on_calendar(d, cal, D[0], D[4])
    assert out["pnl"].to_list() == [0.0, 1.0, 0.0, 2.0, 0.0]
    with pytest.raises(ValueError, match="not on the trading calendar"):
        on_calendar(d, cal.filter(pl.col("date") != D[3]), D[0], D[4])


def test_final_pnl_is_the_last_mark_times_unit_and_scale():
    y = final_pnl(_trades(), PathSizing(unit="size", scale=10.0))
    assert y.to_dicts() == [{"t0": D[0], "y": 100.0}, {"t0": D[1], "y": -30.0}]
