"""Cross-sectional transforms: every statistic is taken within one date (or one date × group), so a transform never
looks across time and appending later rows changes no earlier value.

    percentile_rank   (rank − 1) / (count − 1) within the date, in [0, 1]; nulls stay null
    zscore            (x − mean) / sd within the date
    demean            x − mean within the date × group (sector neutralisation)
    residualize       the residual of a per-date regression on group dummies and exposures (sector and beta
                      neutralisation of a target)

A target with its sector and beta moves left in carries their noise into every IC: on a large-cap panel the
date-to-date IC volatility runs three to four times the independent-names level. Residualizing the target removes
the part a book can hedge; whatever IC survives is what a sector- and beta-neutral book earns.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
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


def residualize(
    frame: pl.DataFrame,
    y: str,
    date: str = "t0",
    groups: Sequence[str] = (),
    exposures: Sequence[str] = (),
    name: str | None = None,
) -> pl.DataFrame:
    """`frame` with `name` (default `<y>_resid`): on each date, `y` minus its least-squares fit on an intercept, one
    dummy per level of each `groups` column and the `exposures` (e.g. a beta estimated from past returns). Uses only
    that date's rows, so it never looks across time. Rows with a null in `y` or an exposure get a null residual;
    a null group is its own level."""
    out_name = name or f"{y}_resid"
    keyed = frame.with_row_index("_row")
    pieces: list[pl.DataFrame] = []
    for (_d,), part in keyed.group_by([date], maintain_order=True):
        ok = part.filter(pl.all_horizontal(pl.col(c).is_not_null() for c in (y, *exposures)))
        cols = [np.ones(ok.height)]
        for g in groups:
            levels = ok[g].cast(pl.String).fill_null("<null>").to_numpy()
            for lv in np.unique(levels)[1:]:  # the first level is the intercept's
                cols.append((levels == lv).astype(float))
        cols += [ok[e].cast(pl.Float64).to_numpy() for e in exposures]
        X, v = np.column_stack(cols), ok[y].cast(pl.Float64).to_numpy()
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        pieces.append(ok.select("_row").with_columns(pl.Series(out_name, v - X @ beta)))
    res = pl.concat(pieces) if pieces else pl.DataFrame(schema={"_row": pl.UInt32, out_name: pl.Float64})
    return keyed.join(res, on="_row", how="left").sort("_row").drop("_row")
