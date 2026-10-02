"""Factor attribution: is a book's return explained by books already known?

    r_t = α + Σ_k β_k · f_k,t + ε_t          (OLS on the dates every series has)

The standard errors are Newey-West (Bartlett kernel, `lags` lags), because overlapping or autocorrelated returns make
OLS errors too small. α is reported per period and annualised (× `periods`); its t-statistic is the incremental test:
a book whose α is indistinguishable from zero adds nothing the factors do not already deliver.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def newey_west(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """The HAC covariance of OLS coefficients."""
    n = X.shape[0]
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    meat = u.T @ u
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        G = u[lag:].T @ u[:-lag]
        meat += w * (G + G.T)
    return XtX_inv @ meat @ XtX_inv * n / max(n - X.shape[1], 1)


def factor_attribution(
    series: Mapping[str, float], factors: Mapping[str, Mapping[str, float]], periods: int = 12, lags: int = 3
) -> dict[str, Any]:
    """Regress `series` on factor return series (dates in common): α per period and annualised with its Newey-West
    t-statistic, betas and their t-statistics, R², the factors' correlation, and the Sharpe of α + residual."""
    names = list(factors)
    common = set(series)
    for f in factors.values():
        common &= set(f)
    dates = sorted(common)
    if len(dates) < len(names) + 5:
        return {"n_obs": len(dates), "reason": "too few common observations"}
    y = np.array([series[d] for d in dates], dtype=float)
    F = np.array([[factors[k][d] for k in names] for d in dates], dtype=float)
    X = np.column_stack([np.ones(len(dates)), F])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    se = np.sqrt(np.diag(newey_west(X, resid, lags)))
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else 0.0
    raw_sharpe = float(y.mean() / y.std(ddof=1) * np.sqrt(periods)) if y.std(ddof=1) > 0 else 0.0
    resid_sharpe = (
        float((beta[0] + resid).mean() / resid.std(ddof=1) * np.sqrt(periods)) if resid.std(ddof=1) > 0 else 0.0
    )
    return {
        "n_obs": len(dates),
        "alpha_per_period": float(beta[0]),
        "alpha_annual": float(beta[0] * periods),
        "alpha_t": float(beta[0] / se[0]) if se[0] > 0 else None,
        "betas": {
            k: {"beta": float(b), "t": float(b / s) if s > 0 else None}
            for k, b, s in zip(names, beta[1:], se[1:], strict=True)
        },
        "r2": float(r2),
        "raw_sharpe": raw_sharpe,
        "alpha_sharpe": resid_sharpe,  # the information ratio of the part the factors do not explain
        "factor_corr": {k: float(np.corrcoef(y, F[:, i])[0, 1]) for i, k in enumerate(names)},
    }
