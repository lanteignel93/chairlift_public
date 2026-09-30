"""The rebuild rules of the design page, as tests: a change re-runs exactly the stages downstream of it."""

import polars as pl
import pytest

from chairlift.core.spec import spec
from chairlift.run.dag import Pipeline, Stage


@spec(name="FeatureSpec")
class FeatureSpec:
    window: int


@spec(name="FoldSpec")
class FoldSpec:
    first_test: int


@spec(name="BookSpec")
class BookSpec:
    quantile: float


RAW = pl.DataFrame({"t": list(range(10)), "x": [float(i % 4) for i in range(10)], "y": [0.1 * i for i in range(10)]})


def build(tmp_path, *, feature_window=3, book_q=0.2, raw_fp="v1", calls=None):
    calls = calls if calls is not None else []

    def src():
        calls.append("sources")
        return RAW

    def feats(s, sources):
        calls.append("features")
        return sources.with_columns(pl.col("x").rolling_mean(s.window, min_samples=1).alias("f"))

    def folds(s, sources):
        calls.append("folds")
        return {"first_test": s.first_test, "n": sources.height}

    def dataset(features):
        calls.append("dataset")
        return features.select("t", "f", "y")

    def fit(dataset, folds):
        calls.append("fit")
        return {"coef": float(dataset["f"].sum()), "first_test": folds["first_test"]}

    def book(s, fit):
        calls.append("book")
        return {"q": s.quantile, "coef": fit["coef"]}

    def report(book):
        calls.append("report")
        return {"line": f"q={book['q']}"}

    stages = [
        Stage("sources", src, fingerprint=lambda: raw_fp),
        Stage("features", feats, ("sources",), FeatureSpec(window=feature_window)),
        Stage("folds", folds, ("sources",), FoldSpec(first_test=5)),
        Stage("dataset", dataset, ("features",)),
        Stage("fit", fit, ("dataset", "folds")),
        Stage("book", book, ("fit",), BookSpec(quantile=book_q)),
        Stage("report", report, ("book",)),
    ]
    return Pipeline(stages, tmp_path), calls


def test_first_run_builds_everything_second_run_hits_everything(tmp_path):
    p, _ = build(tmp_path)
    assert set(p.run().ran()) == set(p.stages)
    p2, calls2 = build(tmp_path)
    r = p2.run()
    assert r.ran() == []
    assert calls2 == []


def test_feature_edit_reruns_downstream_but_not_folds(tmp_path):
    build(tmp_path)[0].run()
    p, _ = build(tmp_path, feature_window=5)
    r = p.run()
    assert set(r.ran()) == {"features", "dataset", "fit", "book", "report"}
    assert r.status()["folds"] == "hit" and r.status()["sources"] == "hit"


def test_book_rule_change_reruns_book_evaluate_report_only(tmp_path):
    build(tmp_path)[0].run()
    p, _ = build(tmp_path, book_q=0.1)
    assert set(p.run().ran()) == {"book", "report"}


def test_changed_source_fingerprint_reruns_everything(tmp_path):
    build(tmp_path)[0].run()
    p, _ = build(tmp_path, raw_fp="v2")
    r = p.run()
    assert r.status()["sources"] == "ran"
    # the source bytes are identical, so downstream keys are unchanged: content addressing absorbs a no-op refresh
    assert r.status()["features"] == "hit"


def test_source_without_fingerprint_is_volatile(tmp_path):
    calls = []
    stages = [Stage("sources", lambda: calls.append(1) or RAW)]
    Pipeline(stages, tmp_path).run()
    Pipeline(stages, tmp_path).run()
    assert len(calls) == 2


def test_missing_object_forces_a_rerun_not_a_silent_skip(tmp_path):
    p, _ = build(tmp_path)
    r = p.run()
    victim = r.results["dataset"].output.split(":")[1]
    next((tmp_path / "store" / "objects").rglob(f"{victim}.*")).unlink()
    p2, _ = build(tmp_path)
    r2 = p2.run()
    assert r2.status()["dataset"] == "ran"
    assert "missing from the store" in r2.results["dataset"].reason


def test_version_bump_reruns_that_stage(tmp_path):
    build(tmp_path)[0].run()
    p, _ = build(tmp_path)
    p.stages["dataset"] = Stage("dataset", p.stages["dataset"].fn, ("features",), version=2)
    r = p.run()
    assert r.status()["dataset"] == "ran" and r.status()["features"] == "hit"


def test_targets_run_only_their_ancestors(tmp_path):
    p, _ = build(tmp_path)
    r = p.run(["dataset"])
    assert set(r.results) == {"sources", "features", "dataset"}


def test_unknown_input_and_cycles_are_rejected(tmp_path):
    with pytest.raises(KeyError):
        Pipeline([Stage("a", lambda b: b, ("b",))], tmp_path)
    with pytest.raises(ValueError):
        Pipeline([Stage("a", lambda b: b, ("b",)), Stage("b", lambda a: a, ("a",))], tmp_path)


def test_manifest_records_reason_and_code_identity(tmp_path):
    p, _ = build(tmp_path)
    p.run(reason="initial build")
    recs = p.manifest.records()
    assert {r.stage for r in recs} == set(p.stages)
    assert all(r.reason == "initial build" and r.code.startswith("chairlift") for r in recs)
