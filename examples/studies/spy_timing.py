"""SPY timing on a synthetic twin: one instrument, daily, a scaled position against buy-and-hold.

    chairlift run examples/studies/spy_timing.py:pipeline
    chairlift run examples/studies/spy_timing.py:pipeline --set beta=0.0     # the null twin: nothing to find

The twin plants a daily drift (the equity premium, found only through the ridge intercept) and small, persistent
predictability from two lagged states among five noise states (`chairlift.verify.twins`).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl

from chairlift.book.timeseries import SignalBook, constant_book, signal_book
from chairlift.evaluate.daily import DailyStatsSpec, daily_stats, paired_sharpe_diff, ts_ic_by_year
from chairlift.learn.fit import FitSpec, fit_walk_forward
from chairlift.learn.models import RidgeSpec
from chairlift.run.dag import Pipeline, Stage
from chairlift.schedule.walkforward import WalkForward, fold_table_json, folds_from_json, make_folds
from chairlift.verify.twins import TimeSeriesTwin, time_series_twin

STUDY_NAME = "spy_timing"


def fit(s: FitSpec, w: WalkForward, twin: pl.DataFrame, folds: dict[str, Any]) -> pl.DataFrame:
    feats = [c for c in twin.columns if c.startswith("x_")]
    return fit_walk_forward(twin, folds_from_json(folds["folds"]), w, feats, s, keys=("date",))[0]


def evaluate(s: DailyStatsSpec, b: SignalBook, twin: pl.DataFrame, preds: pl.DataFrame) -> dict[str, Any]:
    B = signal_book(preds, twin, b)
    hold = constant_book(twin, 1.0, b, B["date"].min(), B["date"].max())  # pyright: ignore[reportArgumentType]
    out = {
        "timing": daily_stats(B["pnl"].to_numpy(), B["date"].to_list(), s),
        "buy_and_hold": daily_stats(hold["pnl"].to_numpy(), hold["date"].to_list(), s),
    }
    out["timing_vs_hold"] = paired_sharpe_diff(hold["pnl"].to_numpy(), B["pnl"].to_numpy(), s)
    out["ic"] = ts_ic_by_year(preds.filter(~pl.col("is_train")).join(twin.select("date", "y"), on="date"))
    return out


def pipeline(root: Path, *, beta: float = 0.05, cost_bp: float = 1.0, seed: int = 11) -> Pipeline:
    twin_spec = TimeSeriesTwin(beta=(beta, beta * 0.6), seed=seed)
    w = WalkForward(start=dt.date(2012, 1, 1), first_test_year=2016, mode="expanding", embargo=1, entry="date")
    ridge = FitSpec(model=RidgeSpec(alpha=1e-2, intercept=True), target="y", rank_target="y", cross_section=False)
    book = SignalBook(cost=cost_bp / 1e4)
    stats = DailyStatsSpec(n_boot=2000, seed=1)
    return Pipeline(
        [
            Stage("twin", time_series_twin, spec=twin_spec, fingerprint=lambda: "synthetic"),
            Stage("folds", lambda w, twin: {"folds": fold_table_json(make_folds(twin, w))}, ("twin",), spec=w),
            Stage("fit_ridge", lambda s, twin, folds: fit(s, w, twin, folds), ("twin", "folds"), spec=ridge),
            Stage(
                "evaluate",
                lambda s, twin, fit_ridge: evaluate(s, book, twin, fit_ridge),
                ("twin", "fit_ridge"),
                spec=stats,
            ),
        ],
        root,
    )
