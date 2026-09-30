"""Walkthrough: how the manifest decides what re-runs.

    uv run python debug_walkthroughs/wt_manifest.py          # asserted run
    uv run python debug_walkthroughs/wt_manifest.py --pdb    # step through

A four-stage toy pipeline (sources → features → dataset → report, plus folds from sources) on ten rows you can hold in
your head. Three builds:
  1. cold: every stage runs;
  2. no change: every stage is a cache hit, and no stage function is called;
  3. the feature window changes 3 → 5: features, dataset and report re-run; sources and folds are hits.

Under --pdb, useful breakpoints (set them with `b <file>:<line>` or by function):
  chairlift/run/dag.py  Pipeline.run         — watch `key`, `rec`, `volatile` for each stage in turn
  chairlift/run/dag.py  Stage.key            — `parts` is exactly what the key hashes: name, version, spec hash,
                                               fingerprint, and one `input=ref` string per input
  chairlift/data/manifest.py Manifest.lookup — `self._by_key` holds one record per key built so far
What to expect: on build 3, `features` gets a new key because its spec hash changed; `dataset` gets a new key because
its input ref changed (different feature bytes); `folds` keeps its key because neither its spec nor its input moved.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import polars as pl

from chairlift.core.spec import spec, spec_hash
from chairlift.run.dag import Pipeline, Stage


@spec(name="WtFeatures")
class FeatureSpec:
    window: int


@spec(name="WtFolds")
class FoldSpec:
    first_test: int


RAW = pl.DataFrame({"t": list(range(10)), "x": [0.0, 1.0, 2.0, 3.0] * 2 + [0.0, 1.0]})
CALLS: list[str] = []


def sources() -> pl.DataFrame:
    CALLS.append("sources")
    return RAW


def features(s: FeatureSpec, sources: pl.DataFrame) -> pl.DataFrame:
    CALLS.append("features")
    return sources.with_columns(pl.col("x").rolling_mean(s.window, min_samples=1).alias("f"))


def folds(s: FoldSpec, sources: pl.DataFrame) -> dict:
    CALLS.append("folds")
    return {"first_test": s.first_test, "rows": sources.height}


def dataset(features: pl.DataFrame) -> pl.DataFrame:
    CALLS.append("dataset")
    return features.select("t", "f")


def report(dataset: pl.DataFrame, folds: dict) -> dict:
    CALLS.append("report")
    return {"mean_f": round(float(dataset["f"].mean()), 6), "first_test": folds["first_test"]}


def pipeline(root: Path, window: int) -> Pipeline:
    return Pipeline(
        [
            Stage("sources", sources, fingerprint=lambda: "raw-v1"),
            Stage("features", features, ("sources",), FeatureSpec(window=window)),
            Stage("folds", folds, ("sources",), FoldSpec(first_test=5)),
            Stage("dataset", dataset, ("features",)),
            Stage("report", report, ("dataset", "folds")),
        ],
        root,
    )


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="wt_manifest_"))
    print(f"store and manifest under {root}\n")

    print("build 1 — cold")
    r1 = pipeline(root, window=3).run(reason="walkthrough build 1")
    print(r1.status())
    assert set(r1.ran()) == {"sources", "features", "folds", "dataset", "report"}
    # rolling mean of x over 3 rows with min_samples=1: t=0..3 → 0, 0.5, 1, 2
    f1 = pipeline(root, window=3).load("dataset", r1)["f"].to_list()
    print("f with window 3:", f1)
    assert f1[:4] == [0.0, 0.5, 1.0, 2.0]

    print("\nbuild 2 — nothing changed")
    CALLS.clear()
    r2 = pipeline(root, window=3).run()
    print(r2.status())
    assert r2.ran() == [] and CALLS == []  # every key is in the manifest and every object in the store

    print("\nbuild 3 — feature window 3 → 5")
    print("feature spec hash 3:", spec_hash(FeatureSpec(window=3))[:12], " 5:", spec_hash(FeatureSpec(window=5))[:12])
    CALLS.clear()
    r3 = pipeline(root, window=5).run(reason="walkthrough: window 5")
    print(r3.status())
    for name, res in r3.results.items():
        print(f"  {name:9s} {res.status:4s} key {res.key[:12]}  {res.reason}")
    assert set(r3.ran()) == {"features", "dataset", "report"}
    assert r3.results["folds"].key == r1.results["folds"].key  # folds depend on sources only
    assert r3.results["dataset"].key != r1.results["dataset"].key  # its input ref changed
    assert "sources" not in CALLS and "folds" not in CALLS

    print("\nmanifest records:", len(pipeline(root, window=5).manifest.records()))
    print("walkthrough passed")


if __name__ == "__main__":
    if "--pdb" in sys.argv:
        import pdb

        pdb.set_trace()  # step into main(): `s` on the next line, then `b chairlift/run/dag.py:<line>` as listed above
    main()
