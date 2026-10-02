"""schedule/walkforward.py: folds whose embargo, window and test year are asserted, on a synthetic calendar."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from chairlift.schedule.walkforward import WalkForward, check_folds, fold_table_json, folds_from_json, make_folds


def index(years: range = range(2016, 2024), hold: int = 21, names: int = 3) -> pl.DataFrame:
    """Business days; every name enters every day and exits `hold` business days later."""
    days = pl.date_range(dt.date(years.start, 1, 4), dt.date(years.stop - 1, 12, 31), "1d", eager=True)
    days = days.filter(days.dt.weekday() <= 5).to_list()
    rows = [(f"N{k}", d, days[min(i + hold, len(days) - 1)], d.year) for i, d in enumerate(days) for k in range(names)]
    return pl.DataFrame(rows, schema=["root", "t0", "exit_date", "y_year"], orient="row")


W = WalkForward(start=dt.date(2016, 1, 4), first_test_year=2017, end=dt.date(2023, 12, 31))


def test_one_fold_per_test_year_expanding_then_rolling():
    folds = make_folds(index(), W)
    assert [f.test_year for f in folds] == list(range(2017, 2024))
    assert [f.mode for f in folds] == ["expanding"] * 4 + ["rolling"] * 3  # 2017-2020 have <= 4 years behind them
    assert folds[4].train_start == dt.date(2017, 1, 1) and folds[-1].train_start == dt.date(2019, 1, 1)


def test_the_embargo_holds_and_is_checked():
    ix = index()
    folds = make_folds(ix, W)
    table = check_folds(ix, folds, W, keys=("root", "t0"))
    assert (table["gap_td"] >= 21).all() and table["gap_td"].min() == 21
    for f in folds:
        tr = ix.filter(f.train_mask(W))
        assert max(tr["exit_date"].to_list()) <= f.train_exit_cut < f.test_start


def test_a_broken_fold_table_is_refused():
    ix = index()
    f = make_folds(ix, W)[0]
    bad = type(f)(f.test_year, f.train_start, f.test_start, f.test_start, f.test_end, f.mode)  # cut at the test start
    with pytest.raises(AssertionError, match=r"embargo|inside the test period"):
        check_folds(ix, [bad], W)


def test_rows_after_end_are_ignored_and_last_year_can_be_cut():
    folds = make_folds(
        index(range(2016, 2026)), WalkForward(start=W.start, first_test_year=2017, end=dt.date(2023, 12, 31))
    )
    assert folds[-1].test_year == 2023
    short = make_folds(index(), WalkForward(start=W.start, first_test_year=2017, last_test_year=2018))
    assert [f.test_year for f in short] == [2017, 2018]


def test_expanding_mode_never_rolls():
    folds = make_folds(index(), WalkForward(start=W.start, first_test_year=2017, mode="expanding"))
    assert {f.train_start for f in folds} == {W.start}


def test_fold_json_round_trip():
    folds = make_folds(index(), W)
    assert folds_from_json(fold_table_json(folds)) == folds
