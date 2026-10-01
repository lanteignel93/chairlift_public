"""Time-series transforms for one instrument (or per instrument with `over`): every value at t uses data up to t only.

rolling_zscore   (x − rolling mean) / rolling sd over the trailing window, t included
lag              x shifted forward by n rows (a feature known at t − n)
forward_return   the log return from t to t + h: a target, never a feature
"""

from __future__ import annotations

import polars as pl


def rolling_zscore(col: str, window: int, min_samples: int | None = None, over: str | None = None) -> pl.Expr:
    x = pl.col(col)
    m = min_samples or window // 2
    mean = x.rolling_mean(window_size=window, min_samples=m)
    sd = x.rolling_std(window_size=window, min_samples=m)
    z = (x - mean) / sd
    return z.over(over) if over else z


def lag(col: str | pl.Expr, n: int = 1, over: str | None = None) -> pl.Expr:
    x = pl.col(col) if isinstance(col, str) else col
    return x.shift(n).over(over) if over else x.shift(n)


def forward_return(price: str, h: int = 1, over: str | None = None) -> pl.Expr:
    r = (pl.col(price).shift(-h) / pl.col(price)).log()
    return r.over(over) if over else r
