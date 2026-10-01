"""A tiny synthetic cross-sectional study, for exercising the runner and the CLI.

50 entities × 300 days, seeded. One feature `x` ~ N(0, 1) per entity-day; the target is y = ic · x + noise, so the
planted within-date correlation between x and y is about `ic`. Stages: sources → features → folds → dataset → fit →
evaluate → report. The fit is a per-fold least-squares slope on training rows only, predicting the test rows.

This is a demonstration pipeline, not a chairlift study: it does not use the study protocols (milestone 0 replaces it
with toy studies built on them).

    uv run chairlift run chairlift.verify.toy:pipeline --root runs/toy
    uv run chairlift run chairlift.verify.toy:pipeline --root runs/toy --set window=5
"""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import polars as pl

from chairlift.core.spec import spec
from chairlift.run import events
from chairlift.run.dag import Pipeline, Stage

STUDY_NAME = "toy"


class Fold(TypedDict):
    train_end: int
    test_start: int
    test_end: int


@spec(name="ToySources")
class SourceSpec:
    entities: int = 50
    days: int = 300
    ic: float = 0.1
    seed: int = 7


@spec(name="ToyFeatures")
class FeatureSpec:
    window: int = 1


@spec(name="ToyFolds")
class FoldSpec:
    first_test: int = 150
    block: int = 50
    embargo: int = 5


def sources(s: SourceSpec) -> pl.DataFrame:
    rng = np.random.default_rng(s.seed)
    n = s.entities * s.days
    x = rng.standard_normal(n)
    y = s.ic * x + np.sqrt(1 - s.ic**2) * rng.standard_normal(n)
    return pl.DataFrame(
        {
            "entity": np.tile(np.arange(s.entities), s.days),
            "t0": np.repeat(np.arange(s.days), s.entities),
            "x": x,
            "y": y,
        }
    )


def features(s: FeatureSpec, sources: pl.DataFrame) -> pl.DataFrame:
    return (
        sources.sort("entity", "t0")
        .with_columns(pl.col("x").rolling_mean(s.window, min_samples=1).over("entity").alias("f"))
        .select("entity", "t0", "f")
    )


def folds(s: FoldSpec, sources: pl.DataFrame) -> dict[str, list[Fold]]:
    last = int(sources["t0"].to_numpy().max())
    out: list[Fold] = []
    start = s.first_test
    while start <= last:
        out.append(
            {"train_end": start - s.embargo - 1, "test_start": start, "test_end": min(start + s.block, last + 1) - 1}
        )
        start += s.block
    return {"folds": out}


def dataset(features: pl.DataFrame, sources: pl.DataFrame) -> pl.DataFrame:
    return features.join(sources.select("entity", "t0", "y"), on=["entity", "t0"])


def fit(dataset: pl.DataFrame, folds: dict[str, list[Fold]], pace: float = 0.0) -> pl.DataFrame:
    preds: list[pl.DataFrame] = []
    n = len(folds["folds"])
    for i, fold in enumerate(folds["folds"]):
        events.progress(
            i, n, f"fold {i + 1} of {n}: train ≤ t{fold['train_end']}, test t{fold['test_start']}–t{fold['test_end']}"
        )
        if pace:
            time.sleep(pace)  # demo only: makes the live view visible on a run that takes milliseconds
        train = dataset.filter(pl.col("t0") <= fold["train_end"])
        test = dataset.filter(pl.col("t0").is_between(fold["test_start"], fold["test_end"]))
        f, y = train["f"].to_numpy(), train["y"].to_numpy()
        slope = float((f * y).sum() / (f * f).sum())
        preds.append(test.select("entity", "t0", "y", (pl.col("f") * slope).alias("pred")).with_columns(fold=pl.lit(i)))
        events.metric("slope", slope, fold=i + 1)
    events.progress(n, n, f"{n} folds fitted")
    return pl.concat(preds)


def evaluate(fit: pl.DataFrame) -> dict[str, Any]:
    ic = (
        fit.group_by("t0")
        .agg(pl.corr("pred", "y", method="spearman").alias("ic"))
        .drop_nulls()
        .sort("t0")["ic"]
        .to_numpy()
    )
    mean, sd = float(ic.mean()), float(ic.std(ddof=1))
    events.metric("oos_ic_mean", mean)
    events.metric("oos_ic_t", mean / sd * np.sqrt(len(ic)))
    return {"days": len(ic), "ic_mean": round(mean, 4), "ic_t": round(mean / sd * np.sqrt(len(ic)), 2)}


def report(evaluate: dict[str, Any]) -> dict[str, str]:
    return {"line": f"OOS IC {evaluate['ic_mean']:+.4f} (t = {evaluate['ic_t']:+.2f}) over {evaluate['days']} days"}


def pipeline(root: Path | str, *, window: int = 1, ic: float = 0.1, seed: int = 7, pace: float = 0.0) -> Pipeline:
    """`pace` (seconds per fold) only slows the fit so the live view can be watched; it is not part of any spec."""
    src_spec = SourceSpec(ic=ic, seed=seed)
    return Pipeline(
        [
            Stage("sources", sources, spec=src_spec, fingerprint=lambda: "synthetic"),
            Stage("features", features, ("sources",), FeatureSpec(window=window)),
            Stage("folds", folds, ("sources",), FoldSpec()),
            Stage("dataset", dataset, ("features", "sources")),
            Stage("fit", functools.partial(fit, pace=pace), ("dataset", "folds")),
            Stage("evaluate", evaluate, ("fit",)),
            Stage("report", report, ("evaluate",)),
        ],
        root,
    )
