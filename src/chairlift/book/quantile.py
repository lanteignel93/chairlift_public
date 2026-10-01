"""The quantile book: every date, rank the out-of-sample prediction among the eligible names; long the top `q`, short
the bottom `q`, one unit per name. Eligibility is the study's (membership, liquidity, events); the book only ranks.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from chairlift.core.spec import spec
from chairlift.features.cross_section import rank_of_rows


@spec(name="QuantileBook", version=1)
class QuantileBook:
    q: float = 0.1
    date: str = "t0"


def quantile_book(
    preds: pl.DataFrame, frame: pl.DataFrame, b: QuantileBook, keys: Sequence[str] = ("root", "t0")
) -> pl.DataFrame:
    """`preds`: keys + pred (+ is_train, out-of-sample rows used); `frame`: keys + `eligible` + anything to carry.

    Returns the eligible rows with `pct` (within-date percentile of pred among the eligible) and `side`."""
    oos = preds.filter(~pl.col("is_train")) if "is_train" in preds.columns else preds
    joined = frame.join(
        oos.select(*keys, "pred", *(["fold"] if "fold" in oos.columns else [])), on=list(keys), how="inner"
    )
    x = joined.filter(pl.col("eligible")).with_columns(
        pct=rank_of_rows("pred", b.date), n_eligible=pl.len().over(b.date)
    )
    side = pl.when(pl.col("pct") >= 1 - b.q).then(pl.lit("long")).when(pl.col("pct") <= b.q).then(pl.lit("short"))
    return x.with_columns(side=side.otherwise(None), decile=(pl.col("pct") * 10).floor().clip(0, 9).cast(pl.Int8))
