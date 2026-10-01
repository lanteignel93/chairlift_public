"""Synthetic twins: data with a planted answer, so a pipeline can be shown to find what is there and nothing else.

Every twin is a pure function of its spec (seeded), so it is also a cacheable source stage.

    cross_section_twin   names × dates; forward returns from a market factor, sector factors and planted signals
    time_series_twin     one instrument; a daily return with planted predictability from lagged states

The cross-section twin also carries a release-lag trap: a fundamental `g` that moves the forward return but is only
published `release_lag` dates after its value date. `f_fund` is what a careful study sees at t0 (the latest
published value, describing t0 − lag): honest, and worth nothing. `f_fund_peek` is the value-date series used as if
known at t0: a look-ahead leak, and predictive. A pipeline that trusts IC from `f_fund_peek` has a look-ahead bug; the
twin makes the size of that bug measurable.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from chairlift.core.spec import spec


@spec(name="CrossSectionTwin", version=1)
class CrossSectionTwin:
    names: int = 200
    dates: int = 144  # monthly: 12 years
    sectors: int = 10
    signals: tuple[float, ...] = (0.06, 0.04)  # planted per-feature rank IC (approximately) with the forward return
    noise_features: int = 6
    market_vol: float = 0.045
    sector_vol: float = 0.02
    idio_vol: float = 0.08
    sector_tilt: float = 0.5  # share of each signal's cross-section that is sector (a neutralisation test)
    fundamental_ic: float = 0.05  # the fundamental's planted IC as of its value date
    release_lag: int = 2  # dates between a fundamental's value date and its publication
    start: dt.date = dt.date(2012, 1, 31)
    step_days: int = 30
    seed: int = 7


def cross_section_twin(s: CrossSectionTwin) -> pl.DataFrame:
    """One row per (root, t0): features known at t0, sector, the forward return y over (t0, exit_date]."""
    rng = np.random.default_rng(s.seed)
    n, T, K = s.names, s.dates, len(s.signals)
    sector = rng.integers(0, s.sectors, n)
    dates = [s.start + dt.timedelta(days=s.step_days * t) for t in range(T + s.release_lag + 1)]
    sig_sector = rng.standard_normal((T, s.sectors, K))
    sig = np.sqrt(s.sector_tilt) * sig_sector[:, sector, :] + np.sqrt(1 - s.sector_tilt) * rng.standard_normal(
        (T, n, K)
    )
    market = s.market_vol * rng.standard_normal(T)
    sect = s.sector_vol * rng.standard_normal((T, s.sectors))
    fund = rng.standard_normal((T, n))
    alpha = ((sig * np.array(s.signals)).sum(axis=2) + s.fundamental_ic * fund) * s.idio_vol
    y = market[:, None] + sect[:, sector] + alpha + s.idio_vol * rng.standard_normal((T, n))
    rows: dict[str, list[object]] = {"root": [], "t0": [], "exit_date": [], "sector": []}
    for t in range(T):
        rows["root"] += [f"N{i:03d}" for i in range(n)]
        rows["t0"] += [dates[t]] * n
        rows["exit_date"] += [dates[t + 1]] * n
        rows["sector"] += [f"S{k}" for k in sector]
    out = pl.DataFrame(rows).with_columns(y=pl.Series(y.ravel()))
    for k in range(K):
        out = out.with_columns(pl.Series(f"f_sig{k}", sig[:, :, k].ravel()))
    for j in range(s.noise_features):
        out = out.with_columns(pl.Series(f"f_noise{j}", rng.standard_normal(T * n)))
    published = np.full((T, n), np.nan)
    published[s.release_lag :] = fund[: T - s.release_lag]
    out = out.with_columns(f_fund=pl.Series(published.ravel()), f_fund_peek=pl.Series(fund.ravel()))
    return out.with_columns(y_year=pl.col("t0").dt.year()).fill_nan(None)


@spec(name="TimeSeriesTwin", version=1)
class TimeSeriesTwin:
    days: int = 3000
    beta: tuple[float, ...] = (0.05, 0.03)  # planted predictive correlation of each lagged state with the next return
    noise_features: int = 5
    persistence: float = 0.97  # AR(1) coefficient of each state
    drift: float = 0.0003  # daily mean return (a carry to be found by an intercept)
    vol: float = 0.012
    start: dt.date = dt.date(2012, 1, 3)
    seed: int = 11


def time_series_twin(s: TimeSeriesTwin) -> pl.DataFrame:
    """One row per business day: states x_k (persistent), next-day return r_next = drift + Σ β_k·vol·x_k + noise."""
    rng = np.random.default_rng(s.seed)
    K = len(s.beta)
    X = np.zeros((s.days, K + s.noise_features))
    e = rng.standard_normal(X.shape) * np.sqrt(1 - s.persistence**2)
    for t in range(1, s.days):
        X[t] = s.persistence * X[t - 1] + e[t]
    b = np.array(s.beta)
    r_next = (
        s.drift + s.vol * (X[:, :K] @ b) + s.vol * np.sqrt(max(1e-9, 1 - (b**2).sum())) * rng.standard_normal(s.days)
    )
    days: list[dt.date] = []
    d = s.start
    while len(days) < s.days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    out = pl.DataFrame({"date": days, "r_next": r_next})
    for k in range(K):
        out = out.with_columns(pl.Series(f"x_sig{k}", X[:, k]))
    for j in range(s.noise_features):
        out = out.with_columns(pl.Series(f"x_noise{j}", X[:, K + j]))
    return out.with_columns(
        y=pl.col("r_next"), exit_date=pl.col("date").shift(-1), y_year=pl.col("date").dt.year()
    ).drop_nulls("exit_date")
