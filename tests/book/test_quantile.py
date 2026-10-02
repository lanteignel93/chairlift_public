"""book/quantile.py: the decile book ranks among eligible names only."""

from __future__ import annotations

import polars as pl
import pytest

from chairlift.book.quantile import QuantileBook, quantile_book


def test_top_and_bottom_decile_among_eligible_names():
    n = 21
    frame = pl.DataFrame(
        {"root": [f"N{i}" for i in range(n)] + ["X"], "t0": [1] * (n + 1), "eligible": [True] * n + [False]}
    )
    preds = pl.DataFrame(
        {
            "root": frame["root"],
            "t0": frame["t0"],
            "pred": [float(i) for i in range(n)] + [999.0],
            "is_train": [False] * (n + 1),
        }
    )
    X = quantile_book(preds, frame, QuantileBook(q=0.1))
    assert "X" not in X["root"].to_list() and X["n_eligible"].unique().to_list() == [n]
    assert X.filter(pl.col("side") == "long")["root"].to_list() == ["N18", "N19", "N20"]
    assert X.filter(pl.col("side") == "short")["root"].to_list() == ["N0", "N1", "N2"]


def test_training_rows_are_never_booked():
    frame = pl.DataFrame({"root": ["A", "B"], "t0": [1, 1], "eligible": [True, True]})
    preds = pl.DataFrame({"root": ["A", "B"], "t0": [1, 1], "pred": [1.0, 2.0], "is_train": [True, True]})
    assert quantile_book(preds, frame, QuantileBook()).height == 0


def test_a_trailing_pool_ranks_against_earlier_rows_only():
    import datetime as dt

    from chairlift.book.quantile import trailing_pct

    d = [dt.date(2020, 1, 1), dt.date(2020, 1, 1), dt.date(2020, 1, 5), dt.date(2020, 3, 1), dt.date(2020, 3, 1)]
    x = pl.DataFrame({"t0": d, "pred": [1.0, 3.0, 2.0, 5.0, 0.0]})
    pct = trailing_pct(x, "t0", pool_days=10, min_pool=2).to_list()
    assert pct[:2] == [None, None]  # nothing before them; same-day rows never rank against each other
    assert pct[2] == 0.5  # 2.0 against {1.0, 3.0}
    assert pct[3:] == [None, None]  # the pool [2020-02-20, 2020-03-01) is empty
    assert trailing_pct(x, "t0", pool_days=90, min_pool=2).to_list()[3:] == [1.0, 0.0]


def test_a_pool_book_keeps_the_old_hash_when_off():
    from chairlift.book.quantile import QuantileBook
    from chairlift.core.spec import canonical

    assert "pool_days" not in canonical(QuantileBook())
    assert canonical(QuantileBook(pool_days=91))["pool_days"] == 91


def test_a_pool_across_a_refit_compares_standardised_predictions():
    import datetime as dt

    from chairlift.book.quantile import QuantileBook, quantile_book

    # fold 2015's model predicts on a scale 100x fold 2014's; its first event shares a pool with 2014's events
    d = [dt.date(2014, 12, 1) + dt.timedelta(days=i) for i in range(10)]
    train = pl.DataFrame(
        {
            "root": [f"t{i}" for i in range(8)],
            "t0": [dt.date(2014, 1, 1)] * 4 + [dt.date(2015, 1, 1)] * 4,
            "pred": [-1.0, 1.0, -1.0, 1.0, -100.0, 100.0, -100.0, 100.0],
            "fold": [2014] * 4 + [2015] * 4,
            "is_train": [True] * 8,
        }
    )
    test = pl.DataFrame(
        {
            "root": [f"e{i}" for i in range(10)],
            "t0": d,
            "pred": [-0.9, -0.5, -0.1, 0.1, 0.5, 0.9, 0.3, -0.3, 0.7, 0.0],
            "fold": [2014] * 9 + [2015],
            "is_train": [False] * 10,
        }
    )
    frame = test.select("root", "t0", eligible=pl.lit(True))
    test2 = test.with_columns(pred=pl.when(pl.col("fold") == 2015).then(5.0).otherwise(pl.col("pred")))
    preds2 = pl.concat([train, test2])  # e9 (fold 2015) predicts 5.0
    raw = quantile_book(preds2, frame, QuantileBook(pool_days=30, min_pool=5, pool_scale="raw"))
    std = quantile_book(preds2, frame, QuantileBook(pool_days=30, min_pool=5))
    # 5.0 is the top of 2014's raw scale, but only 0.05 sd on fold 2015's own scale: the middle of the pool
    assert raw.filter(pl.col("root") == "e9")["pct"][0] == 1.0
    # standardised, 2014's nine events span ±0.78 sd; 0.043 sd is above four of them
    assert std.filter(pl.col("root") == "e9")["pct"][0] == pytest.approx(4 / 9)
