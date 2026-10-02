"""The quantile book: rank each eligible out-of-sample prediction; long the top `q`, short the bottom `q`, one unit per
name. Eligibility is the study's (membership, liquidity, events); the book only ranks.

Two ways to rank:
- within date (the default): among the names eligible that date, for a cross-section that is complete each day
- against a trailing pool (`pool_days` > 0): among the predictions of rows entered in the `pool_days` calendar days
  strictly before, for events that arrive a few a day. Ranking within the entry day is noise on a 3-event day, and
  ranking within the month uses events that have not happened yet; the pool uses only what was known.

A trailing pool spans refits: for `pool_days` after each new fold, it holds the previous model's predictions next to
the new one's, on different scales. With `pool_scale="train"` (the default) each prediction is first standardised by
its own fold's training-row moments, known when that fold starts, so the pool compares like with like. Within one fold
this is a monotone map and changes no rank. Folds without training rows (a baseline that is not fitted) stay raw.
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from chairlift.core.spec import added, spec
from chairlift.features.cross_section import rank_of_rows


@spec(name="QuantileBook", version=1)
class QuantileBook:
    q: float = 0.1
    date: str = "t0"
    pool_days: int = added(0)  # > 0: rank against the trailing pool of earlier rows, not within date
    min_pool: int = added(0)  # with a pool: rows whose pool is smaller stay unranked
    pool_scale: Literal["train", "raw"] = added("train")  # how predictions from different refits share a pool


def trailing_pct(x: pl.DataFrame, date: str, pool_days: int, min_pool: int) -> pl.Series:
    """Each row's mid-rank percentile among the `pred` of rows dated in [date − pool_days, date); null when that pool
    holds fewer than `min_pool` rows (or none), or when the row's own prediction is null. Rows of one date never rank
    against each other. The pool can span a fold boundary: it then mixes two refits' predictions for `pool_days`."""
    order = x.with_row_index("_i").filter(pl.col("pred").is_not_null()).sort(date, "_i")
    t, v = order[date].to_list(), order["pred"].to_numpy().astype(float)
    pct = np.full(len(t), np.nan)
    i = 0
    while i < len(t):
        j = bisect_right(t, t[i])
        pool = np.sort(v[bisect_left(t, t[i] - dt.timedelta(days=pool_days)) : i])
        if len(pool) >= max(min_pool, 1):
            q = v[i:j]
            pct[i:j] = (np.searchsorted(pool, q, "left") + np.searchsorted(pool, q, "right")) / (2 * len(pool))
        i = j
    out = np.full(x.height, np.nan)
    out[order["_i"].to_numpy()] = pct
    return pl.Series("pct", out).fill_nan(None)


def fold_standardised(preds: pl.DataFrame, oos: pl.DataFrame) -> pl.DataFrame:
    """Each out-of-sample prediction minus its fold's training-row mean, over that sd; folds without training rows
    (or with a constant in-sample prediction) are left as they are."""
    if not {"fold", "is_train"} <= set(preds.columns):
        return oos
    m = (
        preds.filter(pl.col("is_train"))
        .group_by("fold")
        .agg(mu=pl.col("pred").mean(), sd=pl.col("pred").std())
        .filter(pl.col("sd") > 0)
    )
    z = (pl.col("pred") - pl.col("mu")) / pl.col("sd")
    return (
        oos.join(m, on="fold", how="left")
        .with_columns(pred=pl.when(pl.col("sd").is_not_null()).then(z).otherwise(pl.col("pred")))
        .drop("mu", "sd")
    )


def quantile_book(
    preds: pl.DataFrame, frame: pl.DataFrame, b: QuantileBook, keys: Sequence[str] = ("root", "t0")
) -> pl.DataFrame:
    """`preds`: keys + pred (+ is_train, out-of-sample rows used); `frame`: keys + `eligible` + anything to carry.

    Returns the eligible rows with `pct` (the percentile of pred: within date, or against the trailing pool) and
    `side`; a row without a percentile (a pool still too small) has no side."""
    oos = preds.filter(~pl.col("is_train")) if "is_train" in preds.columns else preds
    if b.pool_days > 0 and b.pool_scale == "train":
        oos = fold_standardised(preds, oos)
    joined = frame.join(
        oos.select(*keys, "pred", *(["fold"] if "fold" in oos.columns else [])), on=list(keys), how="inner"
    )
    x = joined.filter(pl.col("eligible"))
    if b.pool_days > 0:
        x = x.with_columns(pct=trailing_pct(x, b.date, b.pool_days, b.min_pool), n_eligible=pl.len().over(b.date))
    else:
        x = x.with_columns(pct=rank_of_rows("pred", b.date), n_eligible=pl.len().over(b.date))
    side = pl.when(pl.col("pct") >= 1 - b.q).then(pl.lit("long")).when(pl.col("pct") <= b.q).then(pl.lit("short"))
    return x.with_columns(side=side.otherwise(None), decile=(pl.col("pct") * 10).floor().clip(0, 9).cast(pl.Int8))
