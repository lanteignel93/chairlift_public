"""learn/selection.py: one representative per correlated cluster, robust singletons, scored on training rows."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from chairlift.features.cross_section import percentile_rank, rank_of_rows
from chairlift.learn.selection import ClusterSelection, clusters, score, select

pytest.importorskip("scipy")


def panel(days: int = 120, names: int = 80, seed: int = 1) -> pl.DataFrame:
    """Two noisy copies of a signal (one cluster), a weaker third copy, an independent good feature, pure noise."""
    rng = np.random.default_rng(seed)
    n = days * names
    s = rng.standard_normal(n)
    good = rng.standard_normal(n)
    y = 0.3 * s + 0.3 * good + rng.standard_normal(n)
    raw = pl.DataFrame(
        {
            "t0": np.repeat(np.arange(days), names),
            "y_year": np.repeat(np.arange(days) // 30, names),
            "a1": s + 0.1 * rng.standard_normal(n),
            "a2": s + 0.1 * rng.standard_normal(n),
            "a3": s + 0.6 * rng.standard_normal(n),
            "good": good,
            "noise": rng.standard_normal(n),
            "y": y,
        }
    )
    feats = ["a1", "a2", "a3", "good", "noise"]
    return raw.select("t0", "y_year", *(percentile_rank(f).alias(f) for f in feats), y_rank=rank_of_rows("y"))


def test_correlated_features_cluster_together():
    groups = clusters(panel(), ["a1", "a2", "a3", "good", "noise"], ClusterSelection(cut=0.6))
    big = [sorted(v) for v in groups.values() if len(v) >= 3]
    assert big == [["a1", "a2", "a3"]]


def test_one_member_per_cluster_and_robust_singletons_only():
    chosen, table = select(panel(), ["a1", "a2", "a3", "good", "noise"], ClusterSelection())
    assert "good" in chosen and "noise" not in chosen
    assert len([f for f in chosen if f.startswith("a")]) == 1 and "a3" not in chosen  # the cleanest copy wins
    assert set(table["feature"]) == {"a1", "a2", "a3", "good", "noise"}


def test_score_signs_and_consistency():
    p = panel().with_columns(neg=1 - pl.col("good"))
    s = score(p, "neg")
    assert s.sign == -1.0 and s.ic < 0 and s.cons == 1.0 and s.n_years == 4


def test_the_incumbent_survives_a_near_tie():
    p = panel()
    first, _ = select(p, ["a1", "a2", "a3"], ClusterSelection(min_cluster=3))
    other = "a2" if first == ["a1"] else "a1"
    again, _ = select(p, ["a1", "a2", "a3"], ClusterSelection(min_cluster=3), previous=[other])
    assert again == [other]
