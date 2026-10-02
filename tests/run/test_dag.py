"""The rebuild rules of the design page, as tests: a change re-runs exactly the stages downstream of it."""

from pathlib import Path
from typing import Any

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


def test_rebase_shares_one_store_across_roots(tmp_path):
    p, _ = build(tmp_path / "a")
    shared = tmp_path / "shared"
    first = p.rebase(tmp_path / "study1", shared).run()
    assert set(first.ran()) == set(p.stages)
    second = p.rebase(tmp_path / "study2", shared).run()
    assert second.ran() == []  # a different study root, the same store: everything reused
    assert any(shared.glob("manifest.*.jsonl")) and not (tmp_path / "study2" / "store").exists()


def test_fingerprint_is_called_once_per_run(tmp_path):
    calls = []

    def fp():
        calls.append(1)
        return "v1"

    stages = [Stage("src", lambda: RAW, fingerprint=fp), Stage("next", lambda src: src.head(2), ("src",))]
    Pipeline(stages, tmp_path).run()
    assert len(calls) == 1  # computed once for the run, shared by the signature and every stage key


def test_a_failing_stage_records_nothing_for_itself(tmp_path):
    def boom(sources):
        raise RuntimeError("no")

    stages = [Stage("sources", lambda: RAW, fingerprint=lambda: "v"), Stage("bad", boom, ("sources",))]
    p = Pipeline(stages, tmp_path)
    with pytest.raises(RuntimeError):
        p.run()
    assert {r.stage for r in p.manifest.records()} == {"sources"}  # the failed stage has no record to reuse


def test_plan_and_run_agree_on_a_cold_and_a_warm_store(tmp_path):
    p, _ = build(tmp_path)
    assert {i.status for i in p.plan()} == {"run", "upstream"}
    p.run()
    assert {i.status for i in build(tmp_path)[0].plan()} == {"hit"}


def test_back_to_back_runs_get_distinct_run_ids_and_files(tmp_path):
    p, _ = build(tmp_path)
    ids = {p.run().run_id for _ in range(3)}  # same experiment, well within one second
    assert len(ids) == 3
    assert len(list((tmp_path / "runs").glob("*.events.jsonl"))) == 3
    assert len(list((tmp_path / "runs").glob("*.json"))) == 3


# ---- captured values ------------------------------------------------------------------------------------------------


def test_a_value_captured_by_a_stage_function_is_part_of_its_identity(tmp_path: Path):
    def make(mode: str) -> Pipeline:
        return Pipeline([Stage("src", lambda: {"mode": mode}, fingerprint=lambda: "v")], tmp_path)

    a, b = make("published").run(), make("peek").run()
    assert b.ran() == ["src"] and a.signature != b.signature  # before: the second run was served the first's output
    assert make("peek").run().ran() == []


def test_partial_arguments_count_and_ignore_excludes_them(tmp_path: Path):
    import functools

    def f(x: int, pace: float = 0.0) -> dict[str, int]:
        return {"x": x}

    s1 = Stage("s", functools.partial(f, 1, pace=0.1), fingerprint=lambda: "v", ignore=("pace",))
    s2 = Stage("s", functools.partial(f, 1, pace=9.0), fingerprint=lambda: "v", ignore=("pace",))
    s3 = Stage("s", functools.partial(f, 2, pace=0.1), fingerprint=lambda: "v", ignore=("pace",))
    assert s1.key({}) == s2.key({}) != s3.key({})


def test_capturing_something_unhashable_fails_when_the_pipeline_is_built():
    frame = pl.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="frame"):
        Stage("s", lambda: frame.height, fingerprint=lambda: "v")


def test_stages_that_capture_nothing_keep_their_keys():
    def f() -> int:
        return 1

    assert Stage("s", f).captured_hash == ""


# ---- code identity --------------------------------------------------------------------------------------------------


def _helper_v1() -> int:
    return 1


def _helper_v2() -> int:
    return 2


def test_editing_a_helper_a_stage_calls_changes_the_stage_key():
    from chairlift.run.dag import code_digest

    def stage_a() -> int:
        return _helper_v1()

    def stage_b() -> int:
        return _helper_v2()

    assert code_digest(stage_a) != code_digest(stage_b)  # same body shape, different helper reached
    assert code_digest(lambda: _helper_v1()) != code_digest(lambda: _helper_v2())
    assert code_digest(stage_a) == code_digest(stage_a)


def test_chairlift_functions_a_stage_reaches_are_part_of_the_digest():
    from chairlift.evaluate.daily import daily_stats
    from chairlift.run.dag import code_digest

    def stage() -> object:
        return daily_stats

    def other() -> object:
        return None

    assert code_digest(stage) != code_digest(other)  # daily_stats' source is in stage's digest


def test_library_code_is_not_part_of_the_digest():
    from chairlift.run.dag import code_digest

    def uses_polars() -> pl.DataFrame:
        return pl.DataFrame({"x": [1]})

    assert len(code_digest(uses_polars)) == 64  # polars' own source is not walked (other module)


def test_the_code_digest_does_not_depend_on_the_hash_seed():
    """Set iteration order changes with PYTHONHASHSEED; the digest walked sets (found: dataclass-generated methods
    share one key, and the first visited won)."""
    import subprocess
    import sys

    prog = (
        "from chairlift.run.dag import code_digest\n"
        "from chairlift.learn.fit import FitSpec, fit_walk_forward\n"
        "from chairlift.learn.models import RidgeSpec, build\n"
        "from chairlift.schedule.walkforward import Fold, WalkForward\n"
        "print(code_digest(lambda: (fit_walk_forward, build, FitSpec, RidgeSpec, Fold, WalkForward)))"
    )
    out = {
        subprocess.run(
            [sys.executable, "-c", prog], capture_output=True, text=True, env={"PYTHONHASHSEED": str(seed)}
        ).stdout
        for seed in range(6)
    }
    digest = out.pop().strip() if len(out) == 1 else ""
    assert (
        len(digest) == 64 and digest != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )  # not empty


def _module_stage(tmp_path: Path, sub: str, text: str) -> Any:
    import importlib.util

    d = tmp_path / sub
    d.mkdir()
    (d / "study_mod.py").write_text(text)
    spec_ = importlib.util.spec_from_file_location(f"study_mod_{sub}", d / "study_mod.py")
    assert spec_ is not None and spec_.loader is not None
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod.stage


def test_a_module_constant_the_stage_reads_is_part_of_the_digest(tmp_path: Path):
    """Found by the event-driven client: `K = 1` → `K = 2` in a study module left every key unchanged."""
    from chairlift.run.dag import code_digest

    body = "import datetime as dt\n{}\n\ndef stage():\n    return K, START\n"
    a = _module_stage(tmp_path, "a", body.format("K = 1\nSTART = dt.date(2011, 6, 1)"))
    b = _module_stage(tmp_path, "b", body.format("K = 2\nSTART = dt.date(2011, 6, 1)"))
    c = _module_stage(tmp_path, "c", body.format("K = 1\nSTART = dt.date(2012, 1, 1)"))
    same = _module_stage(tmp_path, "d", body.format("K = 1\nSTART = dt.date(2011, 6, 1)"))
    assert len({code_digest(a), code_digest(b), code_digest(c)}) == 3
    assert code_digest(a) == code_digest(same)


def test_moving_a_function_in_its_file_keeps_the_digest(tmp_path: Path):
    from chairlift.run.dag import code_digest

    body = "def helper():\n    return 1\n\ndef stage():\n    return helper()\n"
    a = _module_stage(tmp_path, "a", body)
    b = _module_stage(tmp_path, "b", "\n\n# a comment above\n\n" + body)
    assert code_digest(a) == code_digest(b)


def test_a_global_that_cannot_be_a_key_is_skipped(tmp_path: Path):
    from chairlift.run.dag import code_digest

    s = _module_stage(
        tmp_path, "a", "import threading\nLOCK = threading.Lock()\n\ndef stage():\n    with LOCK:\n        return 1\n"
    )
    assert len(code_digest(s)) == 64
