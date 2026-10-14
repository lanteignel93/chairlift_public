"""The walk-forward fit: one model per fold, fitted on that fold's training rows only, predicting its test rows.

Per fold:
- training rows are re-checked against the embargo (the fit never trusts the caller's fold table)
- optional in-fold selection chooses the inputs on the training rows (`learn.selection`)
- optional in-fold transforms (`learn.transforms`: winsorize, interactions, PCA) are fitted on the training rows
- nulls are filled with `fill`, the uninformative value: 0.5 for a percentile rank, 0 for a demeaned rank
- the prediction must vary within every test date (a constant prediction cannot rank); with `cross_section=False`
  (one instrument, or events too sparse to rank within a date) it must vary over the test period instead
- the in-sample IC on the training rows is recorded, so |OOS − IS| can be read: the mean of within-date Spearman ICs,
  or for a time series the Spearman correlation over the training rows

Output: one row per (keys, fold, is_train) with `pred`; test rows are the out-of-sample predictions.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from typing import Any, cast

import polars as pl

from chairlift.core.spec import added, spec
from chairlift.learn.models import ModelSpec, build
from chairlift.learn.selection import ClusterSelection, select
from chairlift.learn.transforms import Transform, fit_apply
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
    cross_section: bool = True  # False: pooled rows (one instrument, or sparse events): checks and IC across rows
    allow_constant: bool = False  # a rule may legitimately hold one position for a whole test year
    transforms: tuple[Transform, ...] = added(())  # fitted on training rows, after selection and the null fill


def in_sample_ic(frame: pl.DataFrame, pred: pl.Series, target: str, date: str, cross_section: bool = True) -> float:
    if not cross_section:
        return float(frame.select(target).with_columns(p=pred).select(pl.corr("p", target, method="spearman")).item())
    by_day = (
        frame.select(date, target)
        .with_columns(p=pred)
        .group_by(date)
        .agg(ic=pl.corr("p", target, method="spearman"))
        .sort(date)  # a deterministic summation order for the mean
    )
    return _num(by_day.fill_nan(None)["ic"].mean())


def _fit_fold(
    f: Fold,
    tr: pl.DataFrame,
    te: pl.DataFrame,
    w: WalkForward,
    features: Sequence[str],
    s: FitSpec,
    keys: Sequence[str],
) -> tuple[list[pl.DataFrame], dict[str, Any]]:
    """One fold, on its training and test rows only (module-level, so a spawned worker can run it)."""
    last_entry, last_exit = tr[w.entry].max(), tr[w.exit].max()
    if not (isinstance(last_entry, dt.date) and isinstance(last_exit, dt.date)):
        raise AssertionError(f"fold {f.test_year}: no training rows")
    if not (last_entry < f.test_start and last_exit <= f.train_exit_cut):
        raise AssertionError(f"fold {f.test_year}: training rows cross the embargo")
    inputs = list(features)
    info: dict[str, Any] = {"test_year": f.test_year, "mode": f.mode, "train_rows": tr.height, "test_rows": te.height}
    if s.selection is not None:
        inputs, table = select(tr, features, s.selection, target=s.rank_target)
        info["selection"] = table.filter(pl.col("chosen")).select("feature", "ic", "ir", "cons").to_dicts()
    # rows without a target carry nothing to learn: they are dropped from the fit (still predicted in-sample)
    has_y = tr[s.target].fill_nan(None).is_not_null().to_numpy()
    if not has_y.any():
        raise AssertionError(f"fold {f.test_year}: no training row has a target")
    y = tr[s.target].fill_nan(None).fill_null(0.0).to_numpy()[has_y]
    X_tr = tr.select(inputs).fill_null(s.fill).to_numpy()
    X_te = te.select(inputs).fill_null(s.fill).to_numpy()
    if s.transforms:
        X_tr, X_te, cols, tinfo = fit_apply(s.transforms, X_tr, y, X_te, inputs, fit_rows=has_y)
        info |= {"transforms": tinfo, "model_inputs": cols}
    names = info.get("model_inputs", inputs)
    m = build(s.model).fit(X_tr[has_y], y, names)
    p_te, p_tr = m.predict(X_te), m.predict(X_tr)
    if s.cross_section:
        sd = te.select(w.entry).with_columns(p=pl.Series(p_te)).group_by(w.entry).agg(sd=pl.col("p").std())
        if sd["sd"].null_count() == sd.height:
            raise AssertionError("one row per date: a time series needs FitSpec(cross_section=False)")
        if not _num(sd["sd"].min()) > 0:
            raise AssertionError(f"fold {f.test_year}: the prediction is constant on some test date")
    elif not s.allow_constant and not float(p_te.std()) > 0:
        raise AssertionError(f"fold {f.test_year}: the prediction is constant over the test period")
    info |= {
        "inputs": inputs,
        "n_inputs": len(inputs),
        "is_ic": in_sample_ic(tr, pl.Series(p_tr), s.rank_target, w.entry, s.cross_section),
        "model": m.describe(),
    }
    out = [te.select(keys).with_columns(pred=pl.Series(p_te), fold=pl.lit(f.test_year), is_train=pl.lit(False))]
    if s.keep_train:
        out.append(tr.select(keys).with_columns(pred=pl.Series(p_tr), fold=pl.lit(f.test_year), is_train=pl.lit(True)))
    return out, info


def fit_walk_forward(
    panel: pl.DataFrame,
    folds: Sequence[Fold],
    w: WalkForward,
    features: Sequence[str],
    s: FitSpec,
    keys: Sequence[str] = ("root", "t0"),
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Every fold, in parallel when the run's compute allows (`compute.workers`); results are identical either way,
    because each fold sees only its own rows and they are assembled in fold order."""
    from chairlift.run.compute import workers

    n = min(workers(), len(folds))
    jobs = [(f, panel.filter(f.train_mask(w)), panel.filter(f.test_mask(w))) for f in folds]
    results: list[tuple[list[pl.DataFrame], dict[str, Any]] | None] = [None] * len(jobs)
    if n > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # spawn, never fork: a forked child inherits polars' thread pool in a dead state and hangs
        with ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn")) as pool:
            futs = {pool.submit(_fit_fold, f, tr, te, w, features, s, keys): i for i, (f, tr, te) in enumerate(jobs)}
            for done, fut in enumerate(as_completed(futs), 1):
                results[futs[fut]] = fut.result()
                events.progress(done, len(jobs), f"fold {jobs[futs[fut]][0].test_year} fitted ({n} workers)")
    else:
        for i, (f, tr, te) in enumerate(jobs):
            events.progress(i, len(jobs), f"fold {f.test_year}: train {f.train_start} → exit ≤ {f.train_exit_cut}")
            results[i] = _fit_fold(f, tr, te, w, features, s, keys)
    preds: list[pl.DataFrame] = []
    infos: list[dict[str, Any]] = []
    for r in results:
        assert r is not None
        preds += r[0]
        infos.append(r[1])
        events.metric("is_ic", r[1]["is_ic"], fold=r[1]["test_year"])
        events.metric("n_inputs", float(r[1]["n_inputs"]), fold=r[1]["test_year"])
    events.progress(len(folds), len(folds), f"{len(folds)} folds fitted")
    return pl.concat(preds), infos


def fold_models(infos: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each fold's description as plain JSON (a stage output): what the fold chose and fitted, by test year. NaN and
    infinite values become null; numpy scalars become Python numbers."""

    def plain(v: object) -> Any:
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in cast("dict[object, object]", v).items()}
        if isinstance(v, list | tuple):
            return [plain(x) for x in cast("Sequence[object]", v)]
        if v is None or isinstance(v, bool | str | int):
            return v
        if isinstance(v, float):
            return float(v) if math.isfinite(v) else None
        item = getattr(v, "item", None)  # a numpy scalar
        return plain(item()) if callable(item) else str(v)

    return [cast("dict[str, Any]", plain(i)) for i in infos]


def _weights(info: dict[str, Any]) -> dict[str, float]:
    """One fold's fitted weight per input: a ridge's standardised coefficient, a composite's weight, LightGBM's mean
    gain (unsigned), a rule's chosen input (1)."""
    d = cast("dict[str, Any]", info.get("model") or {})
    names = [str(n) for n in cast("list[Any]", info.get("model_inputs") or info.get("inputs") or [])]
    kind = d.get("kind")
    if kind == "ridge":
        coef = cast("list[float | None]", d["coef_standardised"])
        return {n: float(c) for n, c in zip(names, coef, strict=True) if c is not None}
    if kind == "composite":
        weights = cast("dict[str, float | None]", d["weights"])
        return {str(n): float(c) for n, c in weights.items() if c is not None}
    if kind == "lightgbm":
        gain = cast("list[float | None]", d["gain_importance_mean"])
        return {n: float(c) for n, c in zip(names, gain, strict=True) if c is not None}
    if kind == "rule":
        j = d.get("input", -1)
        return {names[j]: 1.0} if isinstance(j, int) and 0 <= j < len(names) else {}
    return {}


def stability(models: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How stable a walk-forward model is across its folds, input by input: in how many folds the input has a weight,
    its mean and mean absolute weight, and for a signed model the share of folds that agree with the majority sign. A
    weight that flips sign from fold to fold is noise the model fitted, whatever the out-of-sample number says."""
    kinds = {str(cast("dict[str, Any]", m.get("model") or {}).get("kind")) for m in models}
    signed = kinds <= {"ridge", "composite"}
    per = [_weights(m) for m in models]
    names = sorted({n for w in per for n in w})
    rows: list[dict[str, Any]] = []
    for n in names:
        v = [w[n] for w in per if n in w]
        pos, neg = sum(x > 0 for x in v), sum(x < 0 for x in v)
        rows.append(
            {
                "input": n,
                "folds": len(v),
                "mean": sum(v) / len(v),
                "mean_abs": sum(abs(x) for x in v) / len(v),
                "sign_agreement": max(pos, neg) / len(v) if signed and (pos or neg) else None,
            }
        )
    rows.sort(key=lambda r: -r["mean_abs"])  # a list: a stored JSON object would lose this order
    agree = [r["sign_agreement"] for r in rows if r["sign_agreement"] is not None]
    return {
        "kind": ",".join(sorted(kinds)),
        "folds": len(models),
        "is_ic": {str(m.get("test_year")): m.get("is_ic") for m in models},
        "share_inputs_sign_stable": sum(a == 1.0 for a in agree) / len(agree) if agree else None,
        "inputs": rows,
    }


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
