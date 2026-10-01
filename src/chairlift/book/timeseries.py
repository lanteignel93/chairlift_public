"""The signal book: one instrument, a position through time from an out-of-sample prediction.

Per fold, the prediction is scaled by the standard deviation of that fold's in-sample predictions (known before the
test year starts), so the position is in units of "one in-sample sd of conviction", then capped:

    position_t = clip(pred_t / sd_train, −cap, cap)            ("scaled")
    position_t = sign(pred_t)                                   ("sign")

The position taken at t earns the next period's return; a cost is charged on every change of position:

    pnl_t = position_t · r_{t→t+1} − cost · |position_t − position_{t−1}|
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

import polars as pl

from chairlift.core.spec import spec


@spec(name="SignalBook", version=1)
class SignalBook:
    kind: Literal["scaled", "sign"] = "scaled"
    cap: float = 1.0
    cost: float = 0.0  # per unit of position change, in return units (5 bp = 0.0005)
    long_only: bool = False
    short_only: bool = False
    date: str = "date"
    ret: str = "r_next"  # the next-period return the position earns


def signal_book(preds: pl.DataFrame, frame: pl.DataFrame, b: SignalBook) -> pl.DataFrame:
    """`preds`: date, pred, fold, is_train; `frame`: date + the next-period return column. Out-of-sample rows only."""
    scale = preds.filter(pl.col("is_train")).group_by("fold").agg(sd=pl.col("pred").std())
    oos = preds.filter(~pl.col("is_train")).join(scale, on="fold", how="left")
    raw = pl.col("pred").sign() if b.kind == "sign" else (pl.col("pred") / pl.col("sd")).clip(-b.cap, b.cap)
    lo = 0.0 if b.long_only else -b.cap
    hi = 0.0 if b.short_only else b.cap
    x = (
        oos.join(frame.select(b.date, b.ret), on=b.date, how="inner")
        .sort(b.date)
        .with_columns(position=raw.clip(lo, hi))
    )
    turn = (pl.col("position") - pl.col("position").shift(1).fill_null(0.0)).abs()
    return x.with_columns(turnover=turn, pnl=pl.col("position") * pl.col(b.ret) - b.cost * turn)


def constant_book(frame: pl.DataFrame, position: float, b: SignalBook, start: dt.date, end: dt.date) -> pl.DataFrame:
    """A baseline: hold `position` every day of [start, end] (one entry cost on the first day)."""
    x = frame.filter(pl.col(b.date).is_between(start, end)).select(b.date, b.ret).sort(b.date)
    x = x.with_columns(position=pl.lit(position))
    turn = (pl.col("position") - pl.col("position").shift(1).fill_null(0.0)).abs()
    return x.with_columns(turnover=turn, pnl=pl.col("position") * pl.col(b.ret) - b.cost * turn)


def rule_book(frame: pl.DataFrame, position: str, b: SignalBook, start: dt.date, end: dt.date) -> pl.DataFrame:
    """A rule as a book: the position is a column of `frame` known at t (a volatility target, a term-structure
    switch), clipped to the book's bounds, with the same next-return and turnover-cost accounting."""
    lo = 0.0 if b.long_only else -b.cap
    hi = 0.0 if b.short_only else b.cap
    x = frame.filter(pl.col(b.date).is_between(start, end)).select(b.date, b.ret, position).sort(b.date)
    x = x.with_columns(position=pl.col(position).fill_null(0.0).clip(lo, hi))
    turn = (pl.col("position") - pl.col("position").shift(1).fill_null(0.0)).abs()
    return x.select(b.date, b.ret, "position").with_columns(
        turnover=turn, pnl=pl.col("position") * pl.col(b.ret) - b.cost * turn
    )
