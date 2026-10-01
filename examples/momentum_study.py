"""A small but complete study: cross-sectional momentum on daily closes read through a data alias.

chairlift run examples/momentum_study.py:pipeline --set lookback=60     # needs [data] vendor = "..."
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from chairlift.core.spec import spec
from chairlift.data.refs import DataRef, fingerprint_of
from chairlift.run import events
from chairlift.run.dag import Pipeline, Stage

STUDY_NAME = "momentum"  # runs go to <home>/studies/momentum/

PRICES = DataRef(alias="vendor", relpath="daily_prices")  # resolved through [data] vendor = "..."


@spec(name="MomentumFeature", version=1)
class FeatureSpec:
    lookback: int = 20
    skip: int = 1


@spec(name="MomentumBook", version=1)
class BookSpec:
    quantile: float = 0.1


def prices() -> pl.DataFrame:
    return pl.read_parquet(PRICES.path() / "*.parquet")  # columns: date, ticker, close


def features(s: FeatureSpec, prices: pl.DataFrame) -> pl.DataFrame:
    close = pl.col("close")
    return (
        prices.sort("ticker", "date")
        .with_columns(
            mom=(close.shift(s.skip) / close.shift(s.lookback) - 1).over("ticker"),
            fwd=(close.shift(-1) / close - 1).over("ticker"),
        )
        .drop_nulls(["mom", "fwd"])
    )


def book(s: BookSpec, features: pl.DataFrame) -> dict[str, float]:
    events.progress(0, 2, "ranking")  # shows in the live view; never part of a key
    ranked = features.with_columns(pct=(pl.col("mom").rank() / pl.len()).over("date"))
    long = pl.col("fwd").filter(pl.col("pct") > 1 - s.quantile).mean()
    short = pl.col("fwd").filter(pl.col("pct") <= s.quantile).mean()
    daily = ranked.group_by("date").agg((long - short).alias("ls")).drop_nulls().sort("date")["ls"]
    events.progress(1, 2, "scoring")
    mean, sd = float(daily.mean()), float(daily.std())  # pyright: ignore[reportArgumentType]
    sharpe = mean / sd * 252**0.5 if sd > 0 else 0.0
    events.metric("ls_sharpe", sharpe)
    events.progress(2, 2, f"{daily.len()} days")
    return {"days": daily.len(), "ls_mean_bp": round(mean * 1e4, 3), "ls_sharpe": round(sharpe, 3)}


def pipeline(root: Path, *, lookback: int = 20, skip: int = 1, quantile: float = 0.1) -> Pipeline:
    return Pipeline(
        [
            Stage("prices", prices, fingerprint=fingerprint_of(PRICES)),
            Stage("features", features, ("prices",), FeatureSpec(lookback=lookback, skip=skip)),
            Stage("book", book, ("features",), BookSpec(quantile=quantile)),
        ],
        root,
    )
