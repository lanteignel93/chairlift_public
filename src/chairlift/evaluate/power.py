"""Power: what a study can detect, and the test statistic that detects it best.

A null result means "no edge" only if the study could have seen one. Two numbers say whether it could:
- the minimum detectable effect (MDE) for the test window: the smallest mean IC, and the smallest annualised Sharpe,
  that a two-sided test at `alpha` detects with probability `power`
- the Fama-MacBeth IC t-statistic: the per-date rank IC, averaged, with Newey-West errors. For a weak cross-sectional
  signal it uses every name's rank, where a decile book keeps two tails at one unit each, so it needs fewer dates to
  detect the same signal.

    MDE_IC     = (z_{1−α/2} + z_power) · sd(IC_t) · √(NW inflation) / √T
    MDE_Sharpe = (z_{1−α/2} + z_power) / √years

sd(IC_t) is the date-to-date volatility of the IC. Under independent names it is about 1/√N. Factor structure (sector
and style moves) makes it several times larger, which is why it is measured on the panel rather than assumed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any

import numpy as np
import polars as pl

_N = NormalDist()


def _z(alpha: float, power: float) -> float:
    return _N.inv_cdf(1 - alpha / 2) + _N.inv_cdf(power)


def newey_west_var(x: np.ndarray, lags: int) -> float:
    """The long-run variance of x (Bartlett weights): the variance of its mean is this / len(x)."""
    x = np.asarray(x, float) - np.mean(x)
    T = len(x)
    v = float(x @ x) / T
    for k in range(1, min(lags, T - 1) + 1):
        v += 2 * (1 - k / (lags + 1)) * float(x[k:] @ x[:-k]) / T
    return max(v, 0.0)


def per_date_ic(frame: pl.DataFrame, pred: str, target: str, date: str) -> np.ndarray:
    """Spearman IC of `pred` with `target` on each date, in date order (dates with no defined IC dropped)."""
    ic = (
        frame.drop_nulls([pred, target])
        .group_by(date)
        .agg(ic=pl.corr(pred, target, method="spearman"))
        .sort(date)
        .fill_nan(None)
        .drop_nulls("ic")
    )
    return ic["ic"].to_numpy()


def fama_macbeth_ic(
    frame: pl.DataFrame, pred: str = "pred", target: str = "y", date: str = "t0", lags: int = 3
) -> dict[str, Any]:
    """Mean per-date IC with its Newey-West t-statistic. `lags` ≈ the target's overlap in dates (1–3 for
    non-overlapping monthly returns; the horizon for overlapping ones)."""
    ic = per_date_ic(frame, pred, target, date)
    T = len(ic)
    if T < 3:
        return {"n_dates": T, "reason": "fewer than 3 dates with an IC"}
    m, sd = float(ic.mean()), float(ic.std(ddof=1))
    lrv = newey_west_var(ic, lags)
    t = m / math.sqrt(lrv / T) if lrv > 0 else float("nan")
    return {
        "mean_ic": m,
        "sd_ic": sd,
        "t_nw": t,
        "p_two_sided": 2 * (1 - _N.cdf(abs(t))) if math.isfinite(t) else None,
        "n_dates": T,
        "lags": lags,
    }


def mde_ic(sd_ic: float, n_dates: int, alpha: float = 0.01, power: float = 0.8, nw_inflation: float = 1.0) -> float:
    """The smallest mean IC detectable over `n_dates` with this date-to-date IC volatility."""
    return _z(alpha, power) * sd_ic * math.sqrt(nw_inflation) / math.sqrt(max(n_dates, 1))


def mde_sharpe(years: float, alpha: float = 0.01, power: float = 0.8) -> float:
    """The smallest annualised Sharpe a test window of `years` detects (iid returns; fat tails need more)."""
    return _z(alpha, power) / math.sqrt(max(years, 1e-9))


def power_report(
    panel: pl.DataFrame,
    features: Sequence[str],
    target: str,
    date: str,
    test_dates: int,
    test_years: float,
    alpha: float = 0.01,
    power: float = 0.8,
) -> dict[str, Any]:
    """What this design can detect: each feature's date-to-date IC volatility on the panel, the median of them, and
    the MDE in IC and in Sharpe for the test window. Nothing here reads a model's predictions."""
    sds: dict[str, float] = {}
    for f in features:
        ic = per_date_ic(panel, f, target, date)
        if len(ic) > 2:
            sds[f] = float(ic.std(ddof=1))
    names: np.ndarray = panel.group_by(date).len()["len"].to_numpy()
    sd_med = float(np.median(list(sds.values()))) if sds else float("nan")
    iid = 1 / math.sqrt(float(np.median(names))) if len(names) else float("nan")
    return {
        "alpha": alpha,
        "power": power,
        "test_dates": test_dates,
        "test_years": test_years,
        "median_names_per_date": float(np.median(names)) if len(names) else None,
        "sd_ic_by_feature": sds,
        "sd_ic_median": sd_med,
        "sd_ic_if_names_independent": iid,
        "factor_inflation": sd_med / iid if iid > 0 else None,
        "mde_ic": mde_ic(sd_med, test_dates, alpha, power) if sds else None,
        "mde_sharpe": mde_sharpe(test_years, alpha, power),
    }
