"""The pipeline's power function: how often a planted signal of a given IC is detected, on twins built like a real
panel (names, dates, date-to-date IC volatility, fat tails).

    detection_rates(spec, seeds)   share of seeds in which the Fama-MacBeth IC test and the decile L/S test detect
                                   the planted signal `f_sig0` at `alpha` (two-sided), and their false-positive rate
                                   on a noise feature
    power_curve(ics, base, seeds)  the same over a grid of planted ICs

A study's null is informative only where this curve is high at the effect sizes it hoped for.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any

import numpy as np
import polars as pl

from chairlift.evaluate.power import fama_macbeth_ic
from chairlift.verify.twins import CrossSectionTwin, cross_section_twin


def _decile_t(frame: pl.DataFrame, feature: str, q: float = 0.1) -> float:
    """t-statistic of the per-date top-minus-bottom-`q` mean return (one unit per name per side)."""
    x = frame.with_columns(pct=(pl.col(feature).rank().over("t0") - 1) / (pl.len().over("t0") - 1))
    spread = (
        x.group_by("t0")
        .agg(ls=pl.col("y").filter(pl.col("pct") >= 1 - q).mean() - pl.col("y").filter(pl.col("pct") <= q).mean())
        .sort("t0")["ls"]
        .to_numpy()
    )
    sd = spread.std(ddof=1)
    return float(spread.mean() / sd * math.sqrt(len(spread))) if sd > 0 else 0.0


def detection_rates(spec: CrossSectionTwin, seeds: Sequence[int], alpha: float = 0.01) -> dict[str, Any]:
    z = NormalDist().inv_cdf(1 - alpha / 2)
    fm_hit = dec_hit = fm_false = 0
    sd_ic: list[float] = []
    for seed in seeds:
        P = cross_section_twin(dataclasses.replace(spec, seed=seed))
        fm = fama_macbeth_ic(P, "f_sig0", "y", "t0", lags=0)
        fm_hit += abs(fm["t_nw"]) > z
        sd_ic.append(fm["sd_ic"])
        dec_hit += abs(_decile_t(P, "f_sig0")) > z
        fm_false += abs(fama_macbeth_ic(P, "f_noise0", "y", "t0", lags=0)["t_nw"]) > z
    n = len(seeds)
    return {
        "ic": spec.signals[0],
        "names": spec.names,
        "dates": spec.dates,
        "style_vol": spec.style_vol,
        "sd_ic": float(np.mean(sd_ic)),
        "power_ic_test": fm_hit / n,
        "power_decile_test": dec_hit / n,
        "false_positive_rate": fm_false / n,
        "seeds": n,
    }


def power_curve(
    ics: Sequence[float], base: CrossSectionTwin, seeds: Sequence[int], alpha: float = 0.01
) -> list[dict[str, Any]]:
    return [detection_rates(dataclasses.replace(base, signals=(ic,)), seeds, alpha) for ic in ics]
