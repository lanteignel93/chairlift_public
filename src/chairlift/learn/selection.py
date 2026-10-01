"""In-fold feature selection: one representative per cluster of correlated features, plus robust singletons.

On a fold's training rows only:
1. cluster the candidate features by average linkage on 1 − |ρ|, where ρ is the Pearson correlation of within-date
   ranks (a pooled Spearman net of date effects), cut at |ρ| = `cut`;
2. score every feature by its yearly IC against the target rank: mean IC, IR (mean / sd across years) and sign
   consistency (share of years with the majority sign);
3. clusters of `min_cluster` or more: keep the member that maximizes |IC| × consistency; the previous fold's member is
   kept unless beaten by more than `tie_tol` (stability across refits);
4. smaller clusters (singletons): keep a feature if |IR| ≥ `singleton_ir` and consistency ≥ `singleton_cons` over at
   least two years.

The output is the input list for the fold's model and a table of every score, for audit.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from chairlift.core.spec import spec


@spec(name="ClusterSelection", version=1)
class ClusterSelection:
    cut: float = 0.60
    min_cluster: int = 3
    tie_tol: float = 0.10
    singleton_ir: float = 2.0
    singleton_cons: float = 0.75
    sample: int = 300_000
    seed: int = 0
    fill: float = 0.5  # the uninformative rank used for the correlation matrix


@dataclass(frozen=True)
class Score:
    feature: str
    sign: float
    ic: float
    ir: float
    cons: float
    n_years: int

    @property
    def key(self) -> float:
        return abs(self.ic) * self.cons


def clusters(R: pl.DataFrame, features: Sequence[str], s: ClusterSelection) -> dict[int, list[str]]:
    """Feature clusters on the given rows; ids in scipy's fcluster numbering."""
    hierarchy: Any = importlib.import_module("scipy.cluster.hierarchy")  # the `ml` extra; untyped
    distance: Any = importlib.import_module("scipy.spatial.distance")
    sample = (
        R.sample(n=min(s.sample, R.height), seed=s.seed)
        .select(features)
        .fill_null(s.fill)
        .to_numpy()
        .astype(np.float64)
    )
    corr = np.asarray(np.nan_to_num(np.corrcoef(sample, rowvar=False), nan=0.0))
    np.fill_diagonal(corr, 1.0)
    dist = 1 - np.abs(corr)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    z = hierarchy.linkage(distance.squareform(dist, checks=False), method="average")
    labels = np.asarray(hierarchy.fcluster(z, t=1 - s.cut, criterion="distance"))
    out: dict[int, list[str]] = {}
    for f, k in sorted(zip(features, labels.astype(int), strict=True), key=lambda t: (t[1], t[0])):
        out.setdefault(int(k), []).append(f)
    return out


def score(R: pl.DataFrame, feature: str, target: str = "y_rank", date: str = "t0", year: str = "y_year") -> Score:
    ic = (
        R.select(date, year, feature, target)
        .drop_nulls()
        .group_by(date, year)
        .agg(ic=pl.corr(feature, target, method="spearman"))
        .fill_nan(None)
        .drop_nulls("ic")
        .sort(date)  # group_by returns groups in arbitrary order: sort so every mean sums in date order
        .group_by(year, maintain_order=True)
        .agg(pl.col("ic").mean())
        .sort(year)["ic"]
        .to_numpy()
    )
    if len(ic) == 0:
        return Score(feature, 1.0, 0.0, 0.0, 0.0, 0)
    m = float(ic.mean())
    sd = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    ir = m / sd if sd > 0 else abs(m)
    cons = float((np.sign(ic) == np.sign(m)).mean()) if m != 0 else 0.0
    return Score(feature, float(np.sign(m) or 1.0), m, ir, cons, len(ic))


def select(
    R: pl.DataFrame,
    features: Sequence[str],
    s: ClusterSelection,
    previous: Sequence[str] = (),
    target: str = "y_rank",
) -> tuple[list[str], pl.DataFrame]:
    """(inputs, score table) for one fold; `previous` = the last fold's chosen members (stability tie-break)."""
    groups = clusters(R, features, s)
    big = {k: v for k, v in groups.items() if len(v) >= s.min_cluster}
    singles = [f for v in groups.values() if len(v) < s.min_cluster for f in v]
    prev = set(previous)
    rows: list[dict[str, Any]] = []
    chosen: list[str] = []
    for k, members in big.items():
        sc = {f: score(R, f, target) for f in members}
        best = max(sc.values(), key=lambda p: p.key)
        incumbent = next((sc[f] for f in members if f in prev), None)
        if incumbent is not None and incumbent.key >= (1 - s.tie_tol) * best.key:
            best = incumbent
        chosen.append(best.feature)
        rows += [
            {"cluster": k, "feature": f, "ic": p.ic, "ir": p.ir, "cons": p.cons, "chosen": f == best.feature}
            for f, p in sc.items()
        ]
    for f in singles:
        p = score(R, f, target)
        ok = abs(p.ir) >= s.singleton_ir and p.cons >= s.singleton_cons and p.n_years >= 2
        rows.append({"cluster": -1, "feature": f, "ic": p.ic, "ir": p.ir, "cons": p.cons, "chosen": ok})
        if ok:
            chosen.append(f)
    return chosen, pl.DataFrame(rows)
