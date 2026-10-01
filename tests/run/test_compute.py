"""run/compute.py: parallel folds give the serial result, bit for bit."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from chairlift.learn.fit import FitSpec, fit_walk_forward
from chairlift.learn.models import RidgeSpec
from chairlift.run.compute import bind_compute, workers
from chairlift.schedule.walkforward import WalkForward, make_folds


def test_default_is_serial_and_binding_is_scoped():
    assert workers() == 1
    with bind_compute(4):
        assert workers() == 4
    assert workers() == 1


def test_parallel_folds_equal_serial_folds():
    rng = np.random.default_rng(0)
    days = pl.date_range(dt.date(2016, 1, 4), dt.date(2019, 12, 31), "1d", eager=True)
    days = days.filter(days.dt.weekday() <= 5)
    n = len(days)
    x = rng.standard_normal(n)
    P = (
        pl.DataFrame({"date": days, "x": x, "y": 0.3 * x + rng.standard_normal(n)})
        .with_columns(exit_date=pl.col("date").shift(-1))
        .drop_nulls()
    )
    w = WalkForward(start=dt.date(2016, 1, 4), first_test_year=2017, entry="date", embargo=1, mode="expanding")
    s = FitSpec(model=RidgeSpec(intercept=True), target="y", rank_target="y", cross_section=False)
    serial, _ = fit_walk_forward(P, make_folds(P, w), w, ["x"], s, keys=("date",))
    with bind_compute(3):
        parallel, _ = fit_walk_forward(P, make_folds(P, w), w, ["x"], s, keys=("date",))
    assert serial.equals(parallel)
