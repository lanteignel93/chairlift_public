"""Judging a search: how much of the best candidate's result is the search itself.

walk_forward_selection   each year, take the candidate with the best Sharpe on everything before it; the stitched
                         series is what the search would have delivered, with no hindsight
pbo                      probability of backtest overfitting (Bailey, Borwein, López de Prado and Zhu, 2014):
                         over combinatorially symmetric splits of time into halves, how often the in-sample best
                         candidate ranks below the median out of sample
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _sharpe(x: np.ndarray, periods: int) -> float:
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / sd * math.sqrt(periods)) if sd > 0 else 0.0


def align(series: Mapping[str, Mapping[str, float]]) -> tuple[list[str], list[str], np.ndarray]:
    """(candidates, dates, T × N matrix) on the dates every candidate has."""
    names = list(series)
    common: set[str] = set(series[names[0]]) if names else set()
    for n in names[1:]:
        common &= set(series[n])
    dates: list[str] = sorted(common)
    M = np.array([[series[n][d] for n in names] for d in dates], dtype=float).reshape(len(dates), len(names))
    return names, dates, M


def walk_forward_selection(
    series: Mapping[str, Mapping[str, float]], periods: int = 252, min_history_years: int = 1
) -> dict[str, Any]:
    names, dates, M = align(series)
    years = np.array([int(d[:4]) for d in dates])
    uniq = sorted(set(years.tolist()))
    picks: list[dict[str, Any]] = []
    stitched: list[float] = []
    for y in uniq[min_history_years:]:
        past, now = years < y, years == y
        scores = [_sharpe(M[past, j], periods) for j in range(len(names))]
        j = int(np.argmax(scores))
        picks.append(
            {"year": y, "picked": names[j], "past_sharpe": scores[j], "year_sharpe": _sharpe(M[now, j], periods)}
        )
        stitched += M[now, j].tolist()
    x = np.array(stitched)
    best_hindsight = (
        max(range(len(names)), key=lambda j: _sharpe(M[years >= uniq[min_history_years], j], periods)) if names else 0
    )
    return {
        "picks": picks,
        "selected_oos_sharpe": _sharpe(x, periods),
        "n_days": len(x),
        "best_in_hindsight": names[best_hindsight] if names else None,
        "best_in_hindsight_sharpe": _sharpe(M[years >= uniq[min_history_years], best_hindsight], periods)
        if names
        else None,
    }


def pbo(series: Mapping[str, Mapping[str, float]], blocks: int = 10, periods: int = 252) -> dict[str, Any]:
    names, _, M = align(series)
    n, T = len(names), M.shape[0]
    if n < 2 or blocks * 2 > T:
        return {"pbo": None, "reason": "needs at least two candidates and two observations per block"}
    blocks -= blocks % 2
    edges = [round(T * i / blocks) for i in range(blocks + 1)]
    parts: list[np.ndarray] = [np.arange(edges[i], edges[i + 1]) for i in range(blocks)]
    logits: list[float] = []
    for combo in itertools.combinations(range(blocks), blocks // 2):
        ins = np.concatenate([parts[i] for i in combo])
        out = np.concatenate([parts[i] for i in range(blocks) if i not in combo])
        is_sh = [_sharpe(M[ins, j], periods) for j in range(n)]
        oos_sh = np.array([_sharpe(M[out, j], periods) for j in range(n)])
        best = int(np.argmax(is_sh))
        r = 1 + float((oos_sh < oos_sh[best]).sum()) + 0.5 * float((oos_sh == oos_sh[best]).sum() - 1)  # 1..n
        w = r / (n + 1)  # relative OOS rank of the in-sample winner, in (0, 1)
        logits.append(math.log(w / (1 - w)))
    arr = np.array(logits)
    return {"pbo": float((arr <= 0).mean()), "n_splits": len(arr), "median_logit": float(np.median(arr))}


def summarize(
    series: Mapping[str, Mapping[str, float]], sharpes: Sequence[float], n_obs: int, periods: int = 252
) -> dict[str, Any]:
    from chairlift.evaluate.deflated import deflated_sharpe

    return {
        "deflated": deflated_sharpe(list(sharpes), n_obs=n_obs, periods=periods),
        "walk_forward_selection": walk_forward_selection(series, periods),
        "pbo": pbo(series, periods=periods),
    }
