"""The walk-forward fit: one model per fold, fitted on that fold's training rows only, predicting its test rows.

Per fold:
- training rows are re-checked against the embargo (the fit never trusts the caller's fold table)
- optional in-fold selection chooses the inputs on the training rows (`learn.selection`)
- nulls are filled with `fill`, the uninformative value: 0.5 for a percentile rank, 0 for a demeaned rank
- the prediction must vary within every test date (a constant prediction cannot rank); for a time series
  (`cross_section=False`, one row per date) it must vary over the test period instead
- the in-sample IC on the training rows is recorded, so |OOS − IS| can be read: the mean of within-date Spearman ICs,
  or for a time series the Spearman correlation over the training rows

Output: one row per (keys, fold, is_train) with `pred`; test rows are the out-of-sample predictions.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import polars as pl

from chairlift.core.spec import spec
from chairlift.learn.models import ModelSpec, build
from chairlift.learn.selection import ClusterSelection, select
from chairlift.run import events
from chairlift.schedule.walkforward import Fold, WalkForward


def _num(v: object) -> float:
    return float(v) if isinstance(v, int | float) else float("nan")


@spec(name="WalkForwardFit", version=1)
class FitSpec:
    model: ModelSpec
    target: str = "y_z"  # what the model is trained on
    rank_target: str = "y_rank"  # what selection and the in-sample IC read
    fill: float = 0.5
    selection: ClusterSelection | None = None
    keep_train: bool = True  # also return in-sample predictions on training rows
    cross_section: bool = True  # False: one row per date (a single instrument); checks and IC read across time


def in_sample_ic(frame: pl.DataFrame, pred: pl.Series, target: str, date: str, cross_section: bool = True) -> float:
    if not cross_section:
        return float(frame.select(target).with_columns(p=pred).select(pl.corr("p", target, method="spearman")).item())
    by_day = (
        frame.select(date, target).with_columns(p=pred).group_by(date).agg(ic=pl.corr("p", target, method="spearman"))
    )
    return _num(by_day.fill_nan(None)["ic"].mean())


def fit_walk_forward(
    panel: pl.DataFrame,
    folds: Sequence[Fold],
    w: WalkForward,
    features: Sequence[str],
    s: FitSpec,
    keys: Sequence[str] = ("root", "t0"),
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    preds: list[pl.DataFrame] = []
    infos: list[dict[str, Any]] = []
    for i, f in enumerate(folds):
        events.progress(i, len(folds), f"fold {f.test_year}: train {f.train_start} → exit ≤ {f.train_exit_cut}")
        tr = panel.filter(f.train_mask(w))
        te = panel.filter(f.test_mask(w))
        last_entry, last_exit = tr[w.entry].max(), tr[w.exit].max()
        if not (isinstance(last_entry, dt.date) and isinstance(last_exit, dt.date)):
            raise AssertionError(f"fold {f.test_year}: no training rows")
        if not (last_entry < f.test_start and last_exit <= f.train_exit_cut):
            raise AssertionError(f"fold {f.test_year}: training rows cross the embargo")
        inputs = list(features)
        info: dict[str, Any] = {
            "test_year": f.test_year,
            "mode": f.mode,
            "train_rows": tr.height,
            "test_rows": te.height,
        }
        if s.selection is not None:
            inputs, table = select(tr, features, s.selection, target=s.rank_target)
            info["selection"] = table.filter(pl.col("chosen")).select("feature", "ic", "ir", "cons").to_dicts()
        y = tr[s.target].fill_nan(0.0).fill_null(0.0).to_numpy()
        X_tr = tr.select(inputs).fill_null(s.fill).to_numpy()
        X_te = te.select(inputs).fill_null(s.fill).to_numpy()
        m = build(s.model).fit(X_tr, y)
        p_te, p_tr = m.predict(X_te), m.predict(X_tr)
        if s.cross_section:
            sd = te.select(w.entry).with_columns(p=pl.Series(p_te)).group_by(w.entry).agg(sd=pl.col("p").std())
            if sd["sd"].null_count() == sd.height:
                raise AssertionError("one row per date: a time series needs FitSpec(cross_section=False)")
            if not _num(sd["sd"].min()) > 0:
                raise AssertionError(f"fold {f.test_year}: the prediction is constant on some test date")
        elif not float(p_te.std()) > 0:
            raise AssertionError(f"fold {f.test_year}: the prediction is constant over the test period")
        info |= {
            "inputs": inputs,
            "n_inputs": len(inputs),
            "is_ic": in_sample_ic(tr, pl.Series(p_tr), s.rank_target, w.entry, s.cross_section),
            "model": m.describe(),
        }
        events.metric("is_ic", info["is_ic"], fold=f.test_year)
        events.metric("n_inputs", float(len(inputs)), fold=f.test_year)
        out = [te.select(keys).with_columns(pred=pl.Series(p_te), fold=pl.lit(f.test_year), is_train=pl.lit(False))]
        if s.keep_train:
            out.append(
                tr.select(keys).with_columns(pred=pl.Series(p_tr), fold=pl.lit(f.test_year), is_train=pl.lit(True))
            )
        preds += out
        infos.append(info)
    events.progress(len(folds), len(folds), f"{len(folds)} folds fitted")
    return pl.concat(preds), infos


def zscore_ensemble(
    members: Sequence[pl.DataFrame], weights: Sequence[float], date: str = "t0", cross_section: bool = True
) -> pl.DataFrame:
    """Weighted average of each member's standardised prediction; nothing is refit.

    Cross-section: z-scores within (date, fold, is_train), so training and test rows never mix. Time series (one row
    per date): each member is standardised by its fold's training-row mean and sd, known before the test year, because
    z-scoring over the test period would use the test period's own moments. The members must cover the same rows.
    """
    if len(members) != len(weights) or not members:
        raise ValueError("one weight per member")
    keys = [c for c in members[0].columns if c != "pred"]
    zs: list[pl.DataFrame] = []
    for i, P in enumerate(members):
        if cross_section:
            g = [date, "fold", "is_train"]
            z = (pl.col("pred") - pl.col("pred").mean().over(g)) / (pl.col("pred").std().over(g) + 1e-12)
            zs.append(P.select(*keys, z.alias(f"z{i}")))
        else:
            m = P.filter(pl.col("is_train")).group_by("fold").agg(mu=pl.col("pred").mean(), sd=pl.col("pred").std())
            zs.append(
                P.join(m, on="fold", how="left").select(
                    *keys, ((pl.col("pred") - pl.col("mu")) / (pl.col("sd") + 1e-12)).alias(f"z{i}")
                )
            )
    out = zs[0]
    for z in zs[1:]:
        out = out.join(z, on=keys, how="inner")
    if out.height != members[0].height:
        raise AssertionError("ensemble members do not cover the same rows")
    combo = sum((wt * pl.col(f"z{i}") for i, wt in enumerate(weights)), pl.lit(0.0))
    return out.select(*keys, combo.alias("pred"))
