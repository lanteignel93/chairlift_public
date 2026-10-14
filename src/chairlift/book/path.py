"""The path book: overlapping trades, each marked along its own path, sized by a model's prediction at entry.

A trade enters at `entry` and is marked every day until it exits (a stop, an expiry): one row per (trade, date) with
its cumulative P&L per unit since entry. The first row is the P&L from the entry price to that mark, so a cost paid at
entry lands on the entry day. The book's P&L on a day is the sum, over live trades, of each one's daily change ×
`unit` × `scale` × its weight. The weight comes from the out-of-sample prediction of the trade's entry:

    constant  w = 1
    rank      w = 2 × the prediction's percentile among every earlier prediction (mean about 1: only timing can win)
    tercile   w = 0 / 1 / 2 by that percentile
    inverse   the prediction is a risk forecast: w = median earlier prediction / this one, clipped to [0, cap]
    direct    the prediction is a position: floored at 0, divided by the mean earlier position, clipped to [0, cap]

"Earlier" is the fold's training predictions, then the test predictions already made (entered on earlier dates), so a
drifting test year cannot raise the mean size. `direct` reads the fold's own history only.

Books are marked on the study's trading calendar, from the first test entry to the last P&L date, at zero on a day
with nothing live. A P&L date missing from the calendar is an error: a book marked on its active days only drops a
gated book's flat days and overstates its Sharpe.
"""

from __future__ import annotations

import bisect
import datetime as dt
from typing import Literal

import numpy as np
import polars as pl

from chairlift.core.spec import spec

Rule = Literal["constant", "rank", "tercile", "inverse", "direct"]


@spec(name="PathSizing", version=1)
class PathSizing:
    rule: Rule = "rank"
    entry: str = "t0"  # the trade's key: its entry date (one trade per entry date)
    date: str = "date"  # the mark date
    pnl: str = "pnl"  # cumulative P&L per unit since entry
    unit: str | None = None  # a per-trade multiplier column in the trades table (a vega sizing), constant on the path
    scale: float = 1.0  # a constant multiplier (a contract multiplier)
    cap: float = 2.0  # the largest weight of `inverse` and `direct` (`rank` and `tercile` stay within [0, 2])


def sizes(preds: pl.DataFrame, s: PathSizing) -> pl.DataFrame:
    """(entry, pred, w) for the out-of-sample rows of `preds` (entry, pred, fold, is_train), in entry order."""
    e = s.entry
    out: list[pl.DataFrame] = []
    past: list[float] = []  # every earlier test prediction, sorted
    for fold in sorted(preds["fold"].unique().to_list()):
        p = preds.filter(pl.col("fold") == fold)
        train = p.filter(pl.col("is_train"))["pred"].to_list()
        test = p.filter(~pl.col("is_train")).sort(e)
        values = test["pred"].to_list()
        if s.rule == "constant":
            w = np.ones(len(values))
        elif s.rule == "direct":
            hist = list(train)
            direct: list[float] = []
            for v in values:
                ref = np.clip(np.asarray(hist, float), 0.0, None)
                m = float(ref.mean()) if len(ref) and ref.mean() > 0 else 1.0
                direct.append(float(np.clip(max(v, 0.0) / m, 0.0, s.cap)))
                hist.append(v)
            w = np.asarray(direct)
        else:
            ref = sorted(train + past)
            pct: list[float] = []
            inverse: list[float] = []
            for v in values:
                pct.append(bisect.bisect_left(ref, v) / len(ref) if ref else 0.5)
                med = float(np.median(ref)) if ref else v
                inverse.append(float(np.clip(med / v, 0.0, s.cap)) if v > 0 and med > 0 else 1.0)
                bisect.insort(ref, v)
            q = np.asarray(pct)
            if s.rule == "rank":
                w = 2 * q
            elif s.rule == "tercile":
                w = np.select([q < 1 / 3, q < 2 / 3], [0.0, 1.0], 2.0)
            else:
                w = np.asarray(inverse)
        for v in values:
            bisect.insort(past, v)
        out.append(test.select(e, "pred").with_columns(w=pl.Series(np.asarray(w, float))))
    return pl.concat(out) if out else pl.DataFrame(schema={e: pl.Date, "pred": pl.Float64, "w": pl.Float64})


def daily_pnl(trades: pl.DataFrame, weights: pl.DataFrame, s: PathSizing, weight: str = "w") -> pl.DataFrame:
    """(date, pnl): over the trades that have a weight, the sum of each one's daily P&L change × unit × scale ×
    weight. A trade without a weight is not in the book."""
    e, d = s.entry, s.date
    t = trades.join(weights.select(e, pl.col(weight).alias("_w")), on=e, how="inner").sort(e, d)
    step = pl.col(s.pnl).diff().over(e).fill_null(pl.col(s.pnl))
    unit = pl.col(s.unit) if s.unit is not None else pl.lit(1.0)
    t = t.with_columns(_x=step * unit * pl.col("_w") * s.scale).sort(d, e)
    return t.group_by(d, maintain_order=True).agg(pnl=pl.col("_x").sum()).rename({d: "date"})


def on_calendar(d: pl.DataFrame, calendar: pl.DataFrame, lo: dt.date, hi: dt.date) -> pl.DataFrame:
    """(date, pnl) on every calendar day in [lo, hi], zero when nothing is live. Refuses a P&L date the calendar does
    not have: the book would lose that day's P&L without a trace."""
    have = set(calendar["date"].to_list())
    missing = sorted(x for x in d["date"].to_list() if x not in have)
    if missing:
        shown = ", ".join(str(x) for x in missing[:5]) + (" …" if len(missing) > 5 else "")
        raise ValueError(f"{len(missing)} P&L date(s) are not on the trading calendar: {shown}")
    cal = calendar.select("date").unique().sort("date").filter(pl.col("date").is_between(lo, hi))
    return cal.join(d.select("date", "pnl"), on="date", how="left").with_columns(pl.col("pnl").fill_null(0.0))


def final_pnl(trades: pl.DataFrame, s: PathSizing) -> pl.DataFrame:
    """(entry, y): each trade's P&L at its last mark × unit × scale (the outcome its sizing was about)."""
    unit = pl.col(s.unit).first() if s.unit is not None else pl.lit(1.0)
    last = trades.sort(s.entry, s.date).group_by(s.entry).agg(y=pl.col(s.pnl).last() * unit * s.scale)
    return last.sort(s.entry)
