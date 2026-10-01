"""book/timeseries.py: positions scaled by in-sample conviction, costs on turnover, constant baselines."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from chairlift.book.timeseries import SignalBook, constant_book, signal_book

D = [dt.date(2020, 1, k) for k in range(1, 6)]
FRAME = pl.DataFrame({"date": D, "r_next": [0.01, -0.02, 0.03, 0.0, 0.01]})
PREDS = pl.concat(
    [
        pl.DataFrame(
            {"date": [dt.date(2019, 1, k) for k in (1, 2, 3)], "pred": [-1.0, 0.0, 1.0], "fold": 2020, "is_train": True}
        ),
        pl.DataFrame({"date": D, "pred": [2.0, -0.5, 0.5, 0.0, -3.0], "fold": 2020, "is_train": False}),
    ]
)


def test_scaled_positions_are_capped_and_pay_the_next_return_minus_turnover_cost():
    B = signal_book(PREDS, FRAME, SignalBook(cost=0.001))
    assert B["position"].to_list() == [1.0, -0.5, 0.5, 0.0, -1.0]  # in-sample sd = 1
    assert B["turnover"].to_list() == [1.0, 1.5, 1.0, 0.5, 1.0]
    assert B["pnl"].to_list() == pytest.approx([0.01 - 0.001, 0.01 - 0.0015, 0.015 - 0.001, -0.0005, -0.01 - 0.001])


def test_sign_and_one_sided_books():
    assert signal_book(PREDS, FRAME, SignalBook(kind="sign"))["position"].to_list() == [1.0, -1.0, 1.0, 0.0, -1.0]
    assert signal_book(PREDS, FRAME, SignalBook(short_only=True))["position"].max() == 0.0
    assert signal_book(PREDS, FRAME, SignalBook(long_only=True))["position"].min() == 0.0


def test_constant_baseline_pays_one_entry_cost():
    B = constant_book(FRAME, -1.0, SignalBook(cost=0.001), D[0], D[-1])
    assert B["turnover"].sum() == 1.0 and B["pnl"][0] == pytest.approx(-0.01 - 0.001)


def test_rule_book_reads_positions_from_a_column_and_clips_to_the_book():
    from chairlift.book.timeseries import rule_book

    f = FRAME.with_columns(pos=pl.Series([-2.0, -0.5, 0.0, None, 1.0]))
    B = rule_book(f, "pos", SignalBook(short_only=True), D[0], D[-1])
    assert B["position"].to_list() == [-1.0, -0.5, 0.0, 0.0, 0.0]
