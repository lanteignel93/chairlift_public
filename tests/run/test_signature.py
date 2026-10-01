"""The run signature: inputs-only identity, the same on any machine, recorded with every run."""

import json
from pathlib import Path

import pytest

from chairlift.verify.toy import pipeline


def test_signature_is_independent_of_the_run_root(tmp_path: Path):
    a = pipeline(tmp_path / "machine_a").signature()
    b = pipeline(tmp_path / "elsewhere" / "machine_b").signature()
    assert a.signature == b.signature
    assert a.plan_keys == b.plan_keys


def test_signature_is_known_before_running_and_unchanged_by_running(tmp_path: Path):
    p = pipeline(tmp_path / "r")
    before = p.signature().signature
    report = p.run()
    assert report.signature == before
    assert p.signature().signature == before  # outputs existing in the store do not move the signature


def test_any_parameter_changes_the_signature_and_only_downstream_plan_keys(tmp_path: Path):
    base = pipeline(tmp_path / "r").signature()
    moved = pipeline(tmp_path / "r", window=5).signature()
    assert moved.signature != base.signature
    changed = {n for n in base.plan_keys if base.plan_keys[n] != moved.plan_keys[n]}
    assert changed == {"features", "dataset", "fit", "evaluate", "report"}  # sources and folds keep their keys


def test_data_fingerprint_is_part_of_the_signature(tmp_path: Path):
    assert (
        pipeline(tmp_path / "r", seed=7).signature().signature != pipeline(tmp_path / "r", seed=8).signature().signature
    )


def test_target_subset_has_its_own_signature(tmp_path: Path):
    p = pipeline(tmp_path / "r")
    assert p.signature(["dataset"]).signature != p.signature().signature
    assert set(p.signature(["dataset"]).plan_keys) == {"sources", "features", "dataset"}


def test_run_record_holds_everything_needed_to_reproduce(tmp_path: Path):
    p = pipeline(tmp_path / "r", window=3)
    report = p.run(reason="record test", meta={"study": "chairlift.verify.toy:pipeline", "params": {"window": 3}})
    rec = json.loads((tmp_path / "r" / "runs" / f"{report.run_id}.json").read_text())
    assert rec["signature"] == report.signature and rec["status"] == "ok"
    assert rec["meta"]["params"] == {"window": 3}
    assert rec["stages"]["features"]["spec"]["window"] == 3
    assert rec["stages"]["sources"]["fingerprint"] == "synthetic"
    assert rec["code"].startswith("chairlift") and "polars" in rec["environment"]["packages"]
    assert set(rec["results"]) == set(p.stages) and rec["finished_at"] >= rec["started_at"]


def test_failed_run_is_recorded_as_failed(tmp_path: Path):
    from chairlift.run.dag import Pipeline, Stage

    def boom() -> dict[str, int]:
        raise RuntimeError("stage exploded")

    p = Pipeline([Stage("src", boom, fingerprint=lambda: "x")], tmp_path / "r")
    with pytest.raises(RuntimeError, match="stage exploded"):
        p.run()
    rec = json.loads(next((tmp_path / "r" / "runs").glob("*.json")).read_text())
    assert rec["status"] == "failed" and "stage exploded" in rec["error"]
