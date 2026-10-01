"""The deflated Sharpe ratio (Bailey and López de Prado, 2014): the probability that the best of N trials has a true
Sharpe above zero, once the expected maximum of N lucky draws is subtracted.

    SR₀  = √V · ((1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N·e)))     expected max of N trial Sharpes under the null
    DSR  = Φ( (SR − SR₀) · √(T − 1) / √(1 − γ₃·SR + (γ₄ − 1)/4 · SR²) )

with Sharpes per observation (not annualised), V the variance of the trials' Sharpes, T observations, γ₃ skewness,
γ₄ kurtosis (3 for a normal), γ the Euler–Mascheroni constant. With one trial SR₀ = 0 and DSR is the probabilistic
Sharpe ratio against zero.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any

EULER = 0.5772156649015329
_N = NormalDist()


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    z1 = _N.inv_cdf(1 - 1 / n_trials)
    z2 = _N.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1 - EULER) * z1 + EULER * z2)


def probabilistic_sharpe(sr: float, n_obs: int, sr0: float = 0.0, skew: float = 0.0, kurt: float = 3.0) -> float:
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    return _N.cdf((sr - sr0) * math.sqrt(max(n_obs - 1, 1)) / denom)


def deflated_sharpe(
    sharpes: Sequence[float], n_obs: int, periods: int = 252, skew: float = 0.0, kurt: float = 3.0
) -> dict[str, Any]:
    """`sharpes`: every trial's annualised Sharpe. Returns the best trial's DSR and the pieces behind it."""
    if not sharpes:
        raise ValueError("no trials")
    per = [s / math.sqrt(periods) for s in sharpes]
    best = max(range(len(per)), key=lambda i: per[i])
    n = len(per)
    mean = sum(per) / n
    var = sum((s - mean) ** 2 for s in per) / (n - 1) if n > 1 else 0.0
    sr0 = expected_max_sharpe(n, var)
    return {
        "n_trials": n,
        "best": best,
        "best_sharpe": sharpes[best],
        "expected_max_sharpe_null": sr0 * math.sqrt(periods),
        "dsr": probabilistic_sharpe(per[best], n_obs, sr0, skew, kurt),
        "psr_vs_zero": probabilistic_sharpe(per[best], n_obs, 0.0, skew, kurt),
    }
