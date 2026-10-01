"""Equity long/short on a synthetic twin: monthly cross-section, sector neutralisation, a decile book.

    chairlift run examples/studies/equity_ls.py:pipeline
    chairlift run examples/studies/equity_ls.py:pipeline --set fundamentals=peek   # the look-ahead leak, on purpose

Every stage is a chairlift component; the only study code is the column lists and the one-period P&L path. The
twin plants two signals with known IC and a fundamental published two months late (`chairlift.verify.twins`).
With `fundamentals="published"` the study uses what was knowable at t0; with `"peek"` it uses the value-date series,
and the OOS IC shows how much a release-lag bug would flatter the book.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Literal

import polars as pl

from chairlift.book.quantile import QuantileBook, quantile_book
from chairlift.evaluate.daily import DailyStatsSpec, book_stats, daily_book, ic_by_year
from chairlift.features.cross_section import demean, percentile_rank, rank_of_rows, zscore
from chairlift.learn.fit import FitSpec, fit_walk_forward, zscore_ensemble
from chairlift.learn.models import LightGBMSpec, RidgeSpec
from chairlift.run.dag import Pipeline, Stage
from chairlift.schedule.walkforward import WalkForward, check_folds, fold_table_json, folds_from_json, make_folds
from chairlift.verify.twins import CrossSectionTwin, cross_section_twin

STUDY_NAME = "equity_ls"
KEYS = ["root", "t0", "exit_date", "y_year", "y", "sector"]


def panel(twin: pl.DataFrame, neutral: bool, fundamentals: str) -> pl.DataFrame:
    fund = "f_fund_peek" if fundamentals == "peek" else "f_fund"
    feats = [c for c in twin.columns if c.startswith(("f_sig", "f_noise"))] + [fund]
    R = twin.select(*KEYS, *(percentile_rank(f).alias(f) for f in feats))
    if neutral:
        R = R.with_columns(demean(f, ["t0", "sector"]).alias(f) for f in feats)
        R = R.with_columns(y_n=demean("y", ["t0", "sector"]))
        return R.with_columns(y_z=zscore("y_n"), y_rank=rank_of_rows("y_n")).drop("y_n")
    return R.with_columns(y_z=zscore("y"), y_rank=rank_of_rows("y"))


def fit(s: FitSpec, w: WalkForward, panel: pl.DataFrame, folds: dict[str, Any]) -> pl.DataFrame:
    feats = [c for c in panel.columns if c.startswith("f_")]
    return fit_walk_forward(panel, folds_from_json(folds["folds"]), w, feats, s)[0]


def evaluate(s: DailyStatsSpec, b: QuantileBook, twin: pl.DataFrame, preds: pl.DataFrame) -> dict[str, Any]:
    frame = twin.select("root", "t0", "exit_date", "y_year", "y", y_rank=rank_of_rows("y")).with_columns(
        eligible=pl.lit(True)
    )
    X = quantile_book(preds, frame, b)
    legs = X.filter(pl.col("side").is_not_null())
    paths = pl.concat(  # one period held: marked at entry (0) and at exit (the period's return)
        [
            legs.select("root", "t0", date=pl.col("t0"), pnl=pl.lit(0.0)),
            legs.select("root", "t0", date=pl.col("exit_date"), pnl=pl.col("y")),
        ]
    )
    day = daily_book(legs, paths)
    out = book_stats(day.with_columns(ls=pl.col("ls_pp")), s)  # per-position: an equal-weight decile spread
    oos = preds.filter(~pl.col("is_train")).select("root", "t0", "pred")
    out["ic"] = ic_by_year(frame.join(oos, on=["root", "t0"]))
    return out


def pipeline(
    root: Path,
    *,
    neutral: bool = True,
    fundamentals: Literal["published", "peek"] = "published",
    signal_ic: float = 0.06,
    seed: int = 7,
) -> Pipeline:
    twin_spec = CrossSectionTwin(signals=(signal_ic, signal_ic * 2 / 3), seed=seed)
    w = WalkForward(start=dt.date(2012, 1, 1), first_test_year=2016, mode="expanding", embargo=1)
    ridge = FitSpec(model=RidgeSpec(alpha=1e-2), fill=0.0 if neutral else 0.5)
    gbm_params = {
        "n_estimators": 200,
        "num_leaves": 8,
        "learning_rate": 0.05,
        "min_child_samples": 200,
        "feature_fraction": 0.8,
    }
    gbm = FitSpec(model=LightGBMSpec(params=gbm_params, seeds=(0, 1, 2), threads=4), fill=0.0 if neutral else 0.5)
    stats = DailyStatsSpec(periods=12, block=3, min_per_year=6, min_per_half=3, n_boot=2000, seed=1)
    book = QuantileBook(q=0.1)
    return Pipeline(
        [
            Stage("twin", cross_section_twin, spec=twin_spec, fingerprint=lambda: "synthetic"),
            Stage("panel", lambda twin: panel(twin, neutral, fundamentals), ("twin",)),
            Stage(
                "folds",
                lambda w, panel: {
                    "folds": fold_table_json(f := make_folds(panel, w)),
                    "n": check_folds(panel, f, w).height,
                },
                ("panel",),
                spec=w,
            ),
            Stage("fit_ridge", lambda s, panel, folds: fit(s, w, panel, folds), ("panel", "folds"), spec=ridge),
            Stage("fit_gbm", lambda s, panel, folds: fit(s, w, panel, folds), ("panel", "folds"), spec=gbm),
            Stage(
                "ensemble",
                lambda fit_ridge, fit_gbm: zscore_ensemble([fit_ridge, fit_gbm], [0.5, 0.5]),
                ("fit_ridge", "fit_gbm"),
            ),
            Stage(
                "eval_ridge",
                lambda s, twin, fit_ridge: evaluate(s, book, twin, fit_ridge),
                ("twin", "fit_ridge"),
                spec=stats,
            ),
            Stage(
                "eval_ensemble",
                lambda s, twin, ensemble: evaluate(s, book, twin, ensemble),
                ("twin", "ensemble"),
                spec=stats,
            ),
        ],
        root,
    )
