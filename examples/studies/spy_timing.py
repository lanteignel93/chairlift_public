"""SPY timing on a synthetic twin, declared as a Study: one instrument, daily, a scaled position against buy-and-hold.

    chairlift run examples/studies/spy_timing.py:pipeline
    chairlift run examples/studies/spy_timing.py:pipeline --set beta=0.0     # the null twin: nothing to find

The twin plants a daily drift (the equity premium, found only through the ridge intercept) and small, persistent
predictability from two lagged states among five noise states (`chairlift.verify.twins`).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from chairlift.book.timeseries import SignalBook
from chairlift.evaluate.daily import DailyStatsSpec
from chairlift.learn.fit import FitSpec
from chairlift.learn.models import RidgeSpec
from chairlift.run.dag import Pipeline
from chairlift.schedule.walkforward import WalkForward
from chairlift.study.build import FeatureSet, Model, Source, Study, TimeSeriesBook
from chairlift.verify.twins import TimeSeriesTwin, time_series_twin

STUDY_NAME = "spy_timing"
HEADLINE = {"stage": "eval_ridge", "sharpe": "book.sharpe", "n_obs": "book.n_days"}
FEATURES = ("x_sig0", "x_sig1", "x_noise0", "x_noise1", "x_noise2", "x_noise3", "x_noise4")


def study(*, beta: float = 0.05, cost_bp: float = 1.0, seed: int = 11) -> Study:
    ridge = FitSpec(model=RidgeSpec(alpha=1e-2, intercept=True), target="y", rank_target="y", cross_section=False)
    return Study(
        name=STUDY_NAME,
        sources=(
            Source(
                "twin",
                time_series_twin,
                spec=TimeSeriesTwin(beta=(beta, beta * 0.6), seed=seed),
                fingerprint=lambda: "synthetic",
            ),
        ),
        index="twin",
        schedule=WalkForward(
            start=dt.date(2012, 1, 1), first_test_year=2016, mode="expanding", embargo=1, entry="date"
        ),
        models=(Model("ridge", ridge, panel="twin", features=FeatureSet(columns=FEATURES)),),
        keys=("date",),
        book=TimeSeriesBook(
            spec=SignalBook(cost=cost_bp / 1e4),
            frame="twin",
            constants=(("buy_and_hold", 1.0),),
            benchmark="buy_and_hold",
        ),
        stats=DailyStatsSpec(n_boot=2000, seed=1),
    )


def pipeline(root: Path, **params: object) -> Pipeline:
    return study(**params).pipeline(root)  # pyright: ignore[reportArgumentType]
