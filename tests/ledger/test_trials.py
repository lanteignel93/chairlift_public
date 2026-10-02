"""ledger/trials.py: one trial per signature, the holdout opened once, on the record."""

from __future__ import annotations

from pathlib import Path

import pytest

from chairlift.ledger.trials import Ledger


def test_a_signature_is_one_trial_however_often_it_runs(tmp_path: Path):
    led = Ledger(tmp_path / "ledger.jsonl")
    assert led.record_trial(signature="a", run_id="r1", params={}, reason="first", headline=None) is not None
    assert led.record_trial(signature="a", run_id="r2", params={}, reason="again", headline=None) is None
    led.record_trial(
        signature="b", run_id="r3", params={"x": 1}, reason="", headline={"sharpe": 1.0, "n_obs": 100, "periods": 252}
    )
    assert [t["signature"] for t in led.trials()] == ["a", "b"]


def test_a_new_report_stage_does_not_charge_the_judged_experiment_again(tmp_path: Path):
    led = Ledger(tmp_path / "ledger.jsonl")
    assert led.record_trial(signature="s1", run_id="r1", params={}, reason="", headline=None, trial_key="k") is not None
    # the same headline stage key under a new signature (a stage was added downstream): not a new trial
    assert led.record_trial(signature="s2", run_id="r2", params={}, reason="", headline=None, trial_key="k") is None
    assert led.record_trial(signature="s3", run_id="r3", params={}, reason="", headline=None, trial_key="k2")
    assert [t["run_id"] for t in led.trials()] == ["r1", "r3"]


def test_the_holdout_opens_once_with_a_reason_and_later_trials_are_flagged(tmp_path: Path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.record_trial(signature="a", run_id="r1", params={}, reason="", headline=None)
    with pytest.raises(ValueError, match="reason"):
        led.open_holdout(reason=" ", by="me")
    e = led.open_holdout(reason="spec frozen", by="me")
    assert e["trials_before"] == 1
    with pytest.raises(PermissionError, match="already opened"):
        led.open_holdout(reason="again", by="me")
    t = led.record_trial(signature="b", run_id="r2", params={}, reason="", headline=None)
    assert t is not None and t["after_holdout_open"] is True


def test_a_torn_line_is_skipped(tmp_path: Path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.record_trial(signature="a", run_id="r1", params={}, reason="", headline=None)
    with led.path.open("a") as fh:
        fh.write('{"kind": "tri')
    assert len(led.trials()) == 1
