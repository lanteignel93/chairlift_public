"""Daily-book evaluation: the P&L of an overlapping book marked every day, and its statistics.

A position (keys, side) carries a cumulative P&L path (keys, date, pnl) from its entry mark (pnl = 0) to its exit.
The book's daily P&L is the sum over live positions of each path's daily increment, long minus short. Sharpe is
mean / sd · √252 of that daily series; its confidence interval is a moving-block bootstrap (blocks of `block` days,
so the autocorrelation of overlapping marks is respected).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from chairlift.core.spec import spec


@spec(name="DailyStats", version=1)
class DailyStatsSpec:
    n_boot: int = 2000
    seed: int = 0
    block: int = 21
    alpha: float = 0.01  # two-sided CI level 1 − alpha
    periods: int = 252  # observations per year: 252 daily, 12 monthly (annualisation)
    min_per_year: int = 60  # a calendar year enters the by-year table with at least this many observations
    min_per_half: int = 40


def daily_book(
    positions: pl.DataFrame, paths: pl.DataFrame, keys: Sequence[str] = ("root", "t0"), end: dt.date | None = None
) -> pl.DataFrame:
    """Per date: long, short (sums over live positions), live counts, per-position means, ls = long − short."""
    P = (
        paths.join(positions.select(*keys, "side"), on=list(keys), how="inner")
        .sort(*keys, "date")
        .with_columns(d_pnl=pl.col("pnl").diff().over(list(keys)).fill_null(0.0))
    )
    is_long, is_short = pl.col("side") == "long", pl.col("side") == "short"
    day = (
        P.group_by("date")
        .agg(
            long=pl.col("d_pnl").filter(is_long).sum(),
            short=pl.col("d_pnl").filter(is_short).sum(),
            n_long=is_long.sum(),
            n_short=is_short.sum(),
        )
        .sort("date")
        .with_columns(
            long_pp=pl.col("long") / pl.col("n_long").clip(lower_bound=1),
            short_pp=pl.col("short") / pl.col("n_short").clip(lower_bound=1),
        )
        .with_columns(ls=pl.col("long") - pl.col("short"), ls_pp=pl.col("long_pp") - pl.col("short_pp"))
    )
    return day.filter(pl.col("date") <= end) if end is not None else day


def _sharpe(x: np.ndarray, periods: int = 252) -> float:
    return float(x.mean() / (x.std(ddof=1) + 1e-12) * np.sqrt(periods))


def daily_stats(x: np.ndarray, dates: Sequence[dt.date], s: DailyStatsSpec) -> dict[str, Any]:
    """A return series judged: Sharpe and mean with block-bootstrap CIs, p(mean ≤ 0), Sharpe by year and half-year
    (with the counts of positive ones), drawdown, and how much the best five periods carry (`top5_share`,
    `sharpe_without_top5`)."""
    x = np.asarray(x, float)
    n = len(x)
    years = np.array([d.year for d in dates])
    rng = np.random.default_rng(s.seed)
    lo, hi = s.alpha / 2, 1 - s.alpha / 2
    nb = int(np.ceil(n / s.block))
    starts = rng.integers(0, n - s.block + 1, size=(s.n_boot, nb))
    idx = (starts[:, :, None] + np.arange(s.block)[None, None, :]).reshape(s.n_boot, -1)[:, :n]
    B = x[idx]
    m_b = B.mean(1)
    sh_b = m_b / (B.std(1, ddof=1) + 1e-12) * np.sqrt(s.periods)
    cum = np.cumsum(x)
    by_year = {
        int(y): _sharpe(x[years == y], s.periods) for y in np.unique(years) if (years == y).sum() >= s.min_per_year
    }
    half = np.array([f"{d.year}H{1 if d.month <= 6 else 2}" for d in dates])
    by_half = {k: _sharpe(x[half == k], s.periods) for k in sorted(set(half)) if (half == k).sum() >= s.min_per_half}
    return {
        "n_days": n,
        "mean_per_day": float(x.mean()),
        "sd_per_day": float(x.std(ddof=1)),
        "sharpe": _sharpe(x, s.periods),
        "sharpe_ci": [float(np.quantile(sh_b, lo)), float(np.quantile(sh_b, hi))],
        "mean_ci": [float(np.quantile(m_b, lo)), float(np.quantile(m_b, hi))],
        "p_mean_le_0": float((m_b <= 0).mean()),
        "total": float(cum[-1]),
        "max_drawdown": float((cum - np.maximum.accumulate(cum)).min()),
        "by_year_sharpe": by_year,
        "years_pos": sum(v > 0 for v in by_year.values()),
        "n_years": len(by_year),
        # the red flag "returns dominated by a few observations": the five best periods' share of the total, and the
        # Sharpe once they are removed
        "top5_share": float(np.sort(x)[-5:].sum() / cum[-1]) if n > 5 and cum[-1] != 0 else None,
        "sharpe_without_top5": _sharpe(np.sort(x)[:-5], s.periods) if n > 10 else None,
        "by_halfyear_sharpe": by_half,
        "halfyears_pos": sum(v > 0 for v in by_half.values()),
        "n_halfyears": len(by_half),
    }


def book_stats(
    day: pl.DataFrame, s: DailyStatsSpec, columns: Sequence[str] = ("long", "short", "ls")
) -> dict[str, Any]:
    dates = day["date"].to_list()
    out: dict[str, Any] = {
        "n_days": day.height,
        "avg_live_long": float(day["n_long"].cast(pl.Float64).to_numpy().mean()),
        "avg_live_short": float(day["n_short"].cast(pl.Float64).to_numpy().mean()),
    }
    for c in columns:
        out[c] = daily_stats(day[c].to_numpy(), dates, s)
    return out


def ic_by_year(
    frame: pl.DataFrame, pred: str = "pred", target: str = "y_rank", date: str = "t0", year: str = "y_year"
) -> dict[str, Any]:
    """Out-of-sample Spearman IC per date, averaged per year."""
    ic = (
        frame.group_by(date, year)
        .agg(ic=pl.corr(pred, target, method="spearman"))
        .fill_nan(None)
        .sort(date)  # group_by returns groups in arbitrary order: sort so every mean sums in date order
        .group_by(year, maintain_order=True)
        .agg(pl.col("ic").mean())
        .sort(year)
    )
    v = ic["ic"].to_numpy()
    return {
        "oos_ic_mean": float(v.mean()),
        "oos_ic_worst_year": float(v.min()),
        "oos_ic_by_year": {int(y): float(c) for y, c in zip(ic[year], v, strict=True)},
    }


def ts_ic_by_year(frame: pl.DataFrame, pred: str = "pred", target: str = "y", date: str = "date") -> dict[str, Any]:
    """A time series' out-of-sample IC: the Spearman correlation of prediction and outcome over each year's rows."""
    f = frame.with_columns(year=pl.col(date).dt.year())
    ic = f.group_by("year").agg(ic=pl.corr(pred, target, method="spearman"), n=pl.len()).sort("year").fill_nan(None)
    v = ic["ic"].drop_nulls().to_numpy()  # a constant prediction (a rule that never switched) has no rank correlation
    return {
        "oos_ic_all": _finite(f.select(pl.corr(pred, target, method="spearman")).item()),
        "oos_ic_mean": float(v.mean()) if len(v) else None,
        "oos_ic_worst_year": float(v.min()) if len(v) else None,
        "oos_ic_by_year": {int(y): _finite(c) for y, c in zip(ic["year"], ic["ic"], strict=True)},
    }


def _finite(v: object) -> float | None:
    return float(v) if isinstance(v, int | float) and np.isfinite(v) else None


def paired_sharpe_diff(x1: np.ndarray, x2: np.ndarray, s: DailyStatsSpec) -> dict[str, Any]:
    """Sharpe(x2) − Sharpe(x1) on the same days, with a paired moving-block bootstrap: the same blocks are drawn for
    both series, so the common component cancels and the interval is that of the difference itself."""
    a, b = np.asarray(x1, float), np.asarray(x2, float)
    if len(a) != len(b):
        raise ValueError("paired series must cover the same days")
    n = len(a)
    rng = np.random.default_rng(s.seed)
    nb = int(np.ceil(n / s.block))
    starts = rng.integers(0, n - s.block + 1, size=(s.n_boot, nb))
    idx = (starts[:, :, None] + np.arange(s.block)[None, None, :]).reshape(s.n_boot, -1)[:, :n]

    def sh(m: np.ndarray) -> np.ndarray:
        return m.mean(1) / (m.std(1, ddof=1) + 1e-12) * np.sqrt(s.periods)

    d = sh(b[idx]) - sh(a[idx])
    lo, hi = s.alpha / 2, 1 - s.alpha / 2
    return {
        "d_sharpe": _sharpe(b, s.periods) - _sharpe(a, s.periods),
        "d_sharpe_ci": [float(np.quantile(d, lo)), float(np.quantile(d, hi))],
        "p_d_le_0": float((d <= 0).mean()),
        "corr": _finite(float(np.corrcoef(a, b)[0, 1])) if a.std() > 0 and b.std() > 0 else None,
    }
