"""learn/fit.py: per-fold fits that see training rows only; the z-score ensemble."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from chairlift.features.cross_section import percentile_rank, rank_of_rows, zscore
from chairlift.learn.fit import FitSpec, fit_walk_forward, zscore_ensemble
from chairlift.learn.models import RidgeSpec
from chairlift.schedule.walkforward import WalkForward, make_folds


def panel(names: int = 40, ic: float = 0.3, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days = pl.date_range(dt.date(2016, 1, 4), dt.date(2019, 12, 31), "1d", eager=True)
    days = days.filter(days.dt.weekday() <= 5).to_list()
    t0 = np.repeat(np.array(days, dtype="datetime64[D]"), names)
    exit_ = np.repeat(
        np.array([days[min(i + 21, len(days) - 1)] for i in range(len(days))], dtype="datetime64[D]"), names
    )
    x = rng.standard_normal(len(t0))
    raw = pl.DataFrame(
        {
            "root": np.tile([f"N{k}" for k in range(names)], len(days)),
            "t0": t0,
            "exit_date": exit_,
            "x": x,
            "noise": rng.standard_normal(len(t0)),
            "y": ic * x + rng.standard_normal(len(t0)),
        }
    )
    raw = raw.with_columns(pl.col("t0").cast(pl.Date), pl.col("exit_date").cast(pl.Date))
    return raw.with_columns(
        y_year=pl.col("t0").dt.year(),
        x=percentile_rank("x"),
        noise=percentile_rank("noise"),
        y_z=zscore("y"),
        y_rank=rank_of_rows("y"),
    )


W = WalkForward(start=dt.date(2016, 1, 4), first_test_year=2017)


def test_out_of_sample_predictions_cover_each_test_year_and_recover_the_signal():
    P = panel()
    folds = make_folds(P, W)
    preds, infos = fit_walk_forward(P, folds, W, ["x", "noise"], FitSpec(model=RidgeSpec()))
    oos = preds.filter(~pl.col("is_train"))
    assert sorted(oos["fold"].unique().to_list()) == [2017, 2018, 2019]
    assert oos.height == P.filter(pl.col("t0").dt.year() >= 2017).height
    j = oos.join(P, on=["root", "t0"])
    ic = np.nanmean(j.group_by("t0").agg(ic=pl.corr("pred", "y_rank", method="spearman"))["ic"].to_numpy())
    assert ic > 0.15
    assert all(i["is_ic"] > 0.15 and i["n_inputs"] == 2 for i in infos)


def test_training_rows_never_reach_past_the_embargo():
    P = panel()
    folds = make_folds(P, W)
    preds, _ = fit_walk_forward(P, folds, W, ["x"], FitSpec(model=RidgeSpec()))
    tr = preds.filter(pl.col("is_train")).join(P.select("root", "t0", "exit_date"), on=["root", "t0"])
    for f in folds:
        assert max(tr.filter(pl.col("fold") == f.test_year)["exit_date"].to_list()) <= f.train_exit_cut


def test_a_constant_prediction_is_refused():
    P = panel().with_columns(x=pl.lit(0.5))
    with pytest.raises(AssertionError, match="constant"):
        fit_walk_forward(P, make_folds(P, W), W, ["x"], FitSpec(model=RidgeSpec()))


def test_zscore_ensemble_is_the_weighted_mean_of_within_date_zscores():
    P = panel()
    folds = make_folds(P, W)
    a, _ = fit_walk_forward(P, folds, W, ["x"], FitSpec(model=RidgeSpec()))
    b, _ = fit_walk_forward(P, folds, W, ["noise"], FitSpec(model=RidgeSpec()))
    e = zscore_ensemble([a, b], [0.5, 0.5])
    assert e.height == a.height
    g = e.filter(~pl.col("is_train")).group_by("t0").agg(m=pl.col("pred").mean())
    assert np.abs(g["m"].to_numpy()).max() < 1e-9
    with pytest.raises(AssertionError, match="same rows"):
        zscore_ensemble([a, b.head(10)], [0.5, 0.5])


def ts_panel(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days = pl.date_range(dt.date(2016, 1, 4), dt.date(2019, 12, 31), "1d", eager=True)
    days = days.filter(days.dt.weekday() <= 5)
    n = len(days)
    x = rng.standard_normal(n)
    return (
        pl.DataFrame({"date": days, "x": x, "y": 0.3 * x + rng.standard_normal(n)})
        .with_columns(exit_date=pl.col("date").shift(-5), y_year=pl.col("date").dt.year())
        .drop_nulls()
    )


def test_a_time_series_fits_with_one_row_per_date_and_its_ensemble_scales_by_training_moments():
    P = ts_panel()
    w = WalkForward(start=dt.date(2016, 1, 4), first_test_year=2017, entry="date", embargo=5, mode="expanding")
    s = FitSpec(model=RidgeSpec(), target="y", rank_target="y", cross_section=False)
    with pytest.raises(AssertionError, match="cross_section=False"):  # one row per date
        fit_walk_forward(
            P, make_folds(P, w), w, ["x"], FitSpec(model=RidgeSpec(), target="y", rank_target="y"), keys=("date",)
        )
    preds, infos = fit_walk_forward(P, make_folds(P, w), w, ["x"], s, keys=("date",))
    assert all(i["is_ic"] > 0.15 for i in infos)
    e = zscore_ensemble([preds, preds], [0.5, 0.5], date="date", cross_section=False)
    tr = e.filter(pl.col("is_train")).group_by("fold").agg(m=pl.col("pred").mean(), sd=pl.col("pred").std())
    assert np.abs(tr["m"].to_numpy()).max() < 1e-9 and np.abs(tr["sd"].to_numpy() - 1).max() < 1e-9
