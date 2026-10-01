"""Cross-sectional transforms: every statistic is taken within one date (or one date × group), so a transform never
looks across time and appending later rows changes no earlier value.

    percentile_rank   (rank − 1) / (count − 1) within the date, in [0, 1]; nulls stay null
    zscore            (x − mean) / sd within the date
    demean            x − mean within the date × group (sector neutralisation)
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def percentile_rank(col: str | pl.Expr, by: str | Sequence[str] = "t0") -> pl.Expr:
    """Average-tie percentile rank. The denominator counts non-null values, so a column with gaps still spans [0, 1]."""
    x = pl.col(col) if isinstance(col, str) else col
    return (x.cast(pl.Float64).rank("average").over(by) - 1) / (x.count().over(by) - 1).clip(lower_bound=1)


def rank_of_rows(col: str | pl.Expr, by: str | Sequence[str] = "t0") -> pl.Expr:
    """Percentile rank with the row count as denominator (a target: no nulls expected)."""
    x = pl.col(col) if isinstance(col, str) else col
    return (x.rank("average").over(by) - 1) / (pl.len().over(by) - 1).clip(lower_bound=1)


def zscore(col: str | pl.Expr, by: str | Sequence[str] = "t0", eps: float = 0.0) -> pl.Expr:
    x = pl.col(col) if isinstance(col, str) else col
    return (x - x.mean().over(by)) / (x.std().over(by) + eps)


def demean(col: str | pl.Expr, by: str | Sequence[str]) -> pl.Expr:
    x = pl.col(col) if isinstance(col, str) else col
    return x - x.mean().over(by)


def rank_panel(frame: pl.DataFrame, features: Sequence[str], keep: Sequence[str], by: str = "t0") -> pl.DataFrame:
    """`keep` columns plus every feature as a within-date percentile rank (Float64)."""
    return frame.select([*keep, *(percentile_rank(f, by).alias(f) for f in features)])
