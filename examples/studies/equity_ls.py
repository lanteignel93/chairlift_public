"""Equity long/short on a synthetic twin, declared as a Study: monthly cross-section, sector neutralisation, a decile
book.

    chairlift run examples/studies/equity_ls.py:pipeline
    chairlift run examples/studies/equity_ls.py:pipeline --set fundamentals=peek   # the look-ahead leak, on purpose

The twin plants two signals with known IC and a fundamental published two months late (`chairlift.verify.twins`).
With `fundamentals="published"` the study uses what was knowable at t0; with `"peek"` it uses the value-date series,
and the OOS IC shows how much a release-lag bug would flatter the book.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import polars as pl

from chairlift.book.quantile import QuantileBook
from chairlift.core.spec import spec
from chairlift.evaluate.daily import DailyStatsSpec
from chairlift.features.cross_section import demean, percentile_rank, rank_of_rows, zscore
from chairlift.learn.fit import FitSpec
from chairlift.learn.models import LightGBMSpec, RidgeSpec
from chairlift.run.dag import Pipeline
from chairlift.schedule.walkforward import WalkForward
from chairlift.study.build import CrossSectionBook, Ensemble, FeatureSet, Model, Source, Study
from chairlift.verify.twins import CrossSectionTwin, cross_section_twin

STUDY_NAME = "equity_ls"
HEADLINE = {"stage": "eval_ensemble", "sharpe": "ls.sharpe", "n_obs": "ls.n_days", "periods": 12}
KEYS = ("root", "t0", "exit_date", "y_year", "y", "sector")


@spec(name="EquityPanel", version=1)
class PanelSpec:
    neutral: bool = True
    fundamentals: Literal["published", "peek"] = "published"


def panel(s: PanelSpec, twin: pl.DataFrame) -> pl.DataFrame:
    fund = "f_fund_peek" if s.fundamentals == "peek" else "f_fund"
    feats = [c for c in twin.columns if c.startswith(("f_sig", "f_noise"))] + [fund]
    R = twin.select(*KEYS, *(percentile_rank(f).alias(f) for f in feats))
    if s.neutral:
        R = R.with_columns(demean(f, ["t0", "sector"]).alias(f) for f in feats)
        R = R.with_columns(y_n=demean("y", ["t0", "sector"]))
        return R.with_columns(y_z=zscore("y_n"), y_rank=rank_of_rows("y_n")).drop("y_n")
    return R.with_columns(y_z=zscore("y"), y_rank=rank_of_rows("y"))


def frame(twin: pl.DataFrame) -> pl.DataFrame:
    """What the book sees: every name eligible, the period's return, its rank (for the IC)."""
    return twin.select("root", "t0", "exit_date", "y_year", "y", y_rank=rank_of_rows("y")).with_columns(
        eligible=pl.lit(True)
    )


def one_period_paths(book: pl.DataFrame) -> pl.DataFrame:
    """Each position is held one period: marked at entry (0) and at exit (the period's return)."""
    legs = book.filter(pl.col("side").is_not_null())
    return pl.concat(
        [
            legs.select("root", "t0", date=pl.col("t0"), pnl=pl.lit(0.0)),
            legs.select("root", "t0", date=pl.col("exit_date"), pnl=pl.col("y")),
        ]
    )


def study(
    *,
    neutral: bool = True,
    fundamentals: Literal["published", "peek"] = "published",
    signal_ic: float = 0.06,
    seed: int = 7,
) -> Study:
    fill = 0.0 if neutral else 0.5
    gbm = {
        "n_estimators": 200,
        "num_leaves": 8,
        "learning_rate": 0.05,
        "min_child_samples": 200,
        "feature_fraction": 0.8,
    }
    features = FeatureSet(exclude=(*KEYS, "y_z", "y_rank"))
    return Study(
        name=STUDY_NAME,
        sources=(
            Source(
                "twin",
                cross_section_twin,
                spec=CrossSectionTwin(signals=(signal_ic, signal_ic * 2 / 3), seed=seed),
                fingerprint=lambda: "synthetic",
            ),
            Source("panel", panel, spec=PanelSpec(neutral=neutral, fundamentals=fundamentals), inputs=("twin",)),
            Source("frame", frame, inputs=("twin",)),
        ),
        index="panel",
        schedule=WalkForward(start=dt.date(2012, 1, 1), first_test_year=2016, mode="expanding", embargo=1),
        models=(
            Model("ridge", FitSpec(model=RidgeSpec(alpha=1e-2), fill=fill), panel="panel", features=features),
            Model(
                "gbm",
                FitSpec(model=LightGBMSpec(params=gbm, seeds=(0, 1, 2), threads=4), fill=fill),
                panel="panel",
                features=features,
            ),
        ),
        ensembles=(Ensemble("ensemble", ("ridge", "gbm")),),
        evaluate=("ridge", "ensemble"),
        book=CrossSectionBook(spec=QuantileBook(q=0.1), frame="frame", paths=one_period_paths, per_position=True),
        stats=DailyStatsSpec(periods=12, block=3, min_per_year=6, min_per_half=3, n_boot=2000, seed=1),
    )


def pipeline(root: Path, **params: object) -> Pipeline:
    return study(**params).pipeline(root)  # pyright: ignore[reportArgumentType]
