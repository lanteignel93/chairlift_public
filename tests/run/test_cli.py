import json
from pathlib import Path

from click.testing import CliRunner

from chairlift.ledger.trials import Ledger
from chairlift.run.cli import main

TOY = "chairlift.verify.toy:pipeline"


def invoke(*args):
    result = CliRunner().invoke(main, list(args), catch_exceptions=False)
    return result


def test_run_then_rerun_reuses_everything(tmp_path):
    root = str(tmp_path / "toy")
    first = invoke("run", TOY, "--root", root, "--json")
    assert first.exit_code == 0
    assert {r["status"] for r in json.loads(first.output)} == {"ran"}
    second = invoke("run", TOY, "--root", root, "--json")
    assert {r["status"] for r in json.loads(second.output)} == {"hit"}


def test_plan_predicts_the_rerun_without_running(tmp_path):
    root = str(tmp_path / "toy")
    invoke("run", TOY, "--root", root)
    before = (tmp_path / "toy" / "manifest.jsonl").read_text()
    out = json.loads(invoke("plan", TOY, "--root", root, "--set", "window=5", "--json").output)
    status = {i["stage"]: i["status"] for i in out}
    assert status["sources"] == "hit" and status["folds"] == "hit"
    assert status["features"] == "run"
    assert all(status[s] == "upstream" for s in ("dataset", "fit", "evaluate", "report"))
    assert (tmp_path / "toy" / "manifest.jsonl").read_text() == before  # plan wrote nothing


def test_set_parses_json_values_and_changes_the_result(tmp_path):
    root = str(tmp_path / "toy")
    base = json.loads(invoke("show", TOY, "evaluate", "--root", root).output)
    null = json.loads(invoke("show", TOY, "evaluate", "--root", root, "--set", "ic=0.0").output)
    assert base["ic_mean"] > 0.05  # planted IC 0.1 recovered out of sample
    assert abs(null["ic_mean"]) < 0.03  # no signal, no IC


def test_show_frame_and_log(tmp_path):
    root = str(tmp_path / "toy")
    shown = invoke("show", TOY, "fit", "--root", root, "--rows", "3")
    assert shown.exit_code == 0 and "rows × 5 columns" in shown.output
    logged = json.loads(invoke("log", "--root", root, "--stage", "fit", "--json").output)
    assert len(logged) == 1 and logged[0]["stage"] == "fit"


def test_file_path_study_reference(tmp_path):
    study = tmp_path / "mystudy.py"
    study.write_text(
        "from chairlift.run.dag import Pipeline, Stage\n"
        "def pipeline(root, n=3):\n"
        "    return Pipeline([Stage('numbers', lambda: {'n': n})], root)\n"
    )
    out = invoke("show", f"{study}:pipeline", "numbers", "--root", str(tmp_path / "r"), "--set", "n=4")
    assert json.loads(out.output) == {"n": 4}


def test_bad_references_are_usage_errors(tmp_path):
    runner = CliRunner()
    assert runner.invoke(main, ["run", "no_colon"]).exit_code == 2
    assert runner.invoke(main, ["run", "chairlift.verify.toy:nope"]).exit_code == 2
    assert runner.invoke(main, ["run", TOY, "--set", "unknown_param=1", "--root", str(tmp_path)]).exit_code == 2
    assert runner.invoke(main, ["show", TOY, "nostage", "--root", str(tmp_path)]).exit_code == 2
    assert runner.invoke(main, ["log", "--root", str(tmp_path / "empty")]).exit_code == 2


def test_signature_runs_and_show(tmp_path):
    root = str(tmp_path / "toy")
    sig = json.loads(invoke("signature", TOY, "--root", root, "--json").output)["signature"]
    invoke("run", TOY, "--root", root, "--reason", "first")
    listed = invoke("runs", "--root", root)
    assert listed.exit_code == 0 and "first" in listed.output
    rec = json.loads(invoke("runs", "--root", root, "show", sig[:10]).output)
    assert rec["signature"] == sig and rec["meta"]["study"] == TOY


def test_live_run_renders_the_stage_table_and_finishes(tmp_path):
    out = invoke("run", TOY, "--root", str(tmp_path / "t"), "--live")
    assert out.exit_code == 0
    for stage in ("sources", "features", "fit", "evaluate", "report"):
        assert stage in out.output
    assert "7 ran, 0 reused" in out.output


def test_watch_once_shows_a_finished_run(tmp_path):
    root = str(tmp_path / "t")
    invoke("run", TOY, "--root", root)
    shown = invoke("watch", "--root", root, "--once")
    assert shown.exit_code == 0 and "ok" in shown.output and "oos_ic_mean" in shown.output


def _status(root):
    return CliRunner().invoke(main, ["status", "--root", root, "--json"])


def test_status_exit_codes_follow_the_latest_run(tmp_path):
    root = tmp_path / "t"
    invoke("run", TOY, "--root", str(root))
    ok = _status(str(root))
    assert ok.exit_code == 0 and json.loads(ok.output)["status"] == "ok"
    rec_path = max((root / "runs").glob("*.json"))
    rec = json.loads(rec_path.read_text())
    rec_path.write_text(json.dumps(rec | {"status": "failed"}))
    assert _status(str(root)).exit_code == 1
    import os

    rec_path.write_text(json.dumps(rec | {"status": "running", "pid": os.getpid()}))
    assert _status(str(root)).exit_code == 2  # this process is alive
    rec_path.write_text(json.dumps(rec | {"status": "running", "pid": 2**22 + 12345}))
    died = _status(str(root))
    assert died.exit_code == 1 and json.loads(died.output)["status"] == "died"


def test_status_and_watch_without_runs_are_usage_errors(tmp_path):
    assert CliRunner().invoke(main, ["status", "--root", str(tmp_path)]).exit_code == 2
    assert CliRunner().invoke(main, ["watch", "--root", str(tmp_path), "--once"]).exit_code == 2


# ---- experiment files and sweeps -----------------------------------------------------------------------------------


def _exp(tmp_path, body: str) -> str:
    p = tmp_path / "exp.toml"
    p.write_text(f'study = "{TOY}"\nreason = "from the file"\n' + body)
    return str(p)


def test_run_an_experiment_file_records_it_and_uses_its_targets(tmp_path):
    root = tmp_path / "toy"
    exp = _exp(tmp_path, '[params]\nwindow = 3\n[run]\ntargets = ["folds"]\n')
    out = json.loads(invoke("run", exp, "--root", str(root), "--json").output)
    assert [r["stage"] for r in out] == ["sources", "folds"]
    rec = json.loads(next((root / "runs").glob("*.json")).read_text())
    assert rec["reason"] == "from the file" and rec["targets"] == ["folds"]
    assert rec["meta"]["params"] == {"window": 3} and rec["meta"]["experiment"]["name"] == "exp"
    assert rec["meta"]["experiment"]["text"].startswith("study = ")


def test_set_overrides_the_file_and_a_cli_signature_matches_the_equivalent_file(tmp_path):
    root = str(tmp_path / "toy")
    exp = _exp(tmp_path, "[params]\nwindow = 3\n")
    from_file = json.loads(invoke("signature", exp, "--root", root, "--set", "window=5", "--json").output)
    from_cli = json.loads(invoke("signature", TOY, "--root", root, "--set", "window=5", "--json").output)
    assert from_file["signature"] == from_cli["signature"]


def test_a_wrong_type_fails_before_any_stage_runs(tmp_path):
    root = tmp_path / "toy"
    out = CliRunner().invoke(main, ["run", _exp(tmp_path, '[params]\nwindow = "3"\n'), "--root", str(root)])
    assert out.exit_code == 2 and "Expected `int`, got `str`" in out.output
    assert not (root / "runs").exists()
    out = CliRunner().invoke(main, ["run", TOY, "--root", str(root), "--set", "lookback=3"])
    assert out.exit_code == 2 and "unknown parameter(s) lookback" in out.output


def test_run_refuses_a_sweep_file(tmp_path):
    out = CliRunner().invoke(main, ["run", _exp(tmp_path, "[sweep]\nwindow = [1, 2]\n"), "--root", str(tmp_path)])
    assert out.exit_code == 2 and "chairlift sweep" in out.output


def test_sweep_runs_each_cell_with_its_own_signature_and_reuses_shared_stages(tmp_path):
    root = str(tmp_path / "toy")
    exp = _exp(tmp_path, "[sweep]\nwindow = [1, 3, 5]\n")
    planned = json.loads(invoke("sweep", exp, "--root", root, "--dry-run", "--json").output)
    assert not (tmp_path / "toy" / "runs").exists()  # a dry run builds nothing
    out = json.loads(invoke("sweep", exp, "--root", root, "--json").output)
    cells = out["cells"]
    assert [c["values"] for c in cells] == [{"window": 1}, {"window": 3}, {"window": 5}]
    assert len({c["signature"] for c in cells}) == 3
    assert [c["signature"] for c in cells] == [c["signature"] for c in planned["cells"]]
    assert cells[0]["reused"] == 0 and all(c["reused"] == 2 for c in cells[1:])  # sources and folds are shared
    recs = [json.loads(p.read_text()) for p in sorted((tmp_path / "toy" / "runs").glob("*.json"))]
    assert {r["meta"]["sweep"]["id"] for r in recs} == {out["sweep_id"]}
    assert sorted(r["meta"]["sweep"]["cell"] for r in recs) == [0, 1, 2]
    again = json.loads(invoke("sweep", exp, "--root", root, "--json").output)
    assert all(c["ran"] == 0 for c in again["cells"])


def test_sweep_validates_every_cell_before_running_any(tmp_path):
    root = tmp_path / "toy"
    out = CliRunner().invoke(main, ["sweep", _exp(tmp_path, '[sweep]\nwindow = [1, "x"]\n'), "--root", str(root)])
    assert out.exit_code == 2 and "window='x'" in out.output
    assert not (root / "runs").exists()


# ---- rerun and compare ---------------------------------------------------------------------------------------------


def _records(root):
    return sorted((json.loads(p.read_text()) for p in (root / "runs").glob("*.json")), key=lambda r: r["run_id"])


def test_rerun_reproduces_a_recorded_run_bit_for_bit_from_scratch(tmp_path):
    root = tmp_path / "toy"
    invoke("run", TOY, "--root", str(root), "--set", "window=3")
    rec = _records(root)[-1]
    into = tmp_path / "again"
    out = invoke("rerun", rec["run_id"][:22], "--root", str(root), "--into", str(into), "--json")
    assert out.exit_code == 0
    res = json.loads(out.output)
    assert res["ok"] and len(res["stages"]) == 7 and all(s["match"] for s in res["stages"])
    again = _records(into)[-1]
    assert again["meta"]["rerun_of"] == rec["run_id"] and set(again["results"]) == set(rec["results"])
    assert {r["status"] for r in again["results"].values()} == {"ran"}  # nothing reused: a fresh root


def test_rerun_fails_when_the_signature_no_longer_matches(tmp_path):
    root = tmp_path / "toy"
    invoke("run", TOY, "--root", str(root))
    rec = _records(root)[-1]
    rec["meta"]["params"] = {"window": 9}  # as if the study definition had moved on
    (root / "runs" / f"{rec['run_id']}.json").write_text(json.dumps(rec))
    out = CliRunner().invoke(main, ["rerun", rec["run_id"], "--root", str(root)])
    assert out.exit_code == 1 and "signature differs" in out.output and "features" in out.output


def test_rerun_warns_on_a_different_environment_and_detects_a_changed_output(tmp_path):
    root = tmp_path / "toy"
    invoke("run", TOY, "--root", str(root))
    rec = _records(root)[-1]
    rec["environment"]["hash"] = "0" * 64
    rec["results"]["report"]["output"] = "json:" + "f" * 64
    (root / "runs" / f"{rec['run_id']}.json").write_text(json.dumps(rec))
    out = CliRunner().invoke(main, ["rerun", rec["run_id"], "--root", str(root)])
    assert out.exit_code == 1 and "environment differs" in out.output and "DIFFERS" in out.output
    assert "NOT reproduced" in out.output


def test_rerun_refuses_a_non_empty_directory(tmp_path):
    root = tmp_path / "toy"
    invoke("run", TOY, "--root", str(root))
    out = CliRunner().invoke(main, ["rerun", _records(root)[-1]["run_id"], "--root", str(root), "--into", str(root)])
    assert out.exit_code == 2 and "not empty" in out.output


def test_compare_names_what_differs_between_two_runs(tmp_path):
    root = tmp_path / "toy"
    invoke("run", TOY, "--root", str(root), "--set", "window=1")
    invoke("run", TOY, "--root", str(root), "--set", "window=3")
    a, b = (r["run_id"] for r in _records(root))
    rows = json.loads(invoke("compare", a, b, "--root", str(root), "--json").output)["rows"]
    differ = {(r["section"], r["key"]) for r in rows if not r["same"]}
    assert ("params", "window") in differ and ("spec", "features") in differ
    assert ("spec", "sources") not in differ and ("output", "sources") not in differ
    assert ("output", "evaluate") in differ
    assert any(s == "metric" for s, _ in differ)  # the fit's slopes and the OOS IC moved
    assert not any(s == "env" for s, _ in differ)  # same process, same environment
    text = invoke("compare", a, b, "--root", str(root)).output
    assert "window" in text and "rows differ" in text
    assert "fit.slope[fold=1]" in text and "fit.slope[fold=3]" in text  # dimensions survive rich markup
    assert text.index("params ") < text.index("spec ") < text.index("output ") < text.index("metric ")


def test_show_records_the_study_so_status_and_runs_can_name_it(tmp_path):
    root = tmp_path / "toy"
    invoke("show", TOY, "report", "--root", str(root))
    rec = _records(root)[-1]
    assert rec["meta"]["name"] == "toy" and rec["reason"] == "show report" and rec["targets"] == ["report"]


def test_user_text_in_tables_is_not_eaten_as_rich_markup(tmp_path):
    root = str(tmp_path / "toy")
    invoke("run", TOY, "--root", root, "--target", "sources", "--reason", "retry [bold] after [fold=1] fix")
    assert "retry [bold] after [fold=1] fix" in invoke("runs", "--root", root).output
    assert "retry [bold] after [fold=1] fix" in invoke("log", "--root", root).output


def test_watch_and_status_follow_a_named_run_not_only_the_newest(tmp_path):
    root = tmp_path / "toy"
    invoke("run", TOY, "--root", str(root), "--target", "sources")
    first = _records(root)[-1]["run_id"]
    invoke("run", TOY, "--root", str(root), "--target", "folds", "--set", "window=2")
    out = CliRunner().invoke(main, ["status", "--root", str(root), "--run", first[:22], "--json"])
    assert out.exit_code == 0 and json.loads(out.output)["run_id"] == first
    shown = invoke("watch", "--root", str(root), "--run", first[:22], "--once").output
    assert "sources" in shown and "folds" not in shown
    assert CliRunner().invoke(main, ["status", "--root", str(root), "--run", "1999"]).exit_code == 2


def test_compare_takes_a_reused_stages_metrics_from_the_run_that_built_it(tmp_path):
    root = tmp_path / "toy"
    invoke("run", TOY, "--root", str(root))
    invoke("run", TOY, "--root", str(root))  # every stage a hit: this run emitted no metric
    a, b = (r["run_id"] for r in _records(root))
    rows = json.loads(invoke("compare", a, b, "--root", str(root), "--json").output)["rows"]
    metrics = [r for r in rows if r["section"] == "metric"]
    assert metrics and all(r["same"] and r["b"] != "—" for r in metrics)


def test_search_runs_every_candidate_charges_the_ledger_and_judges_the_search(tmp_path):
    exp = tmp_path / "search.toml"
    spy = Path(__file__).parents[2] / "examples" / "studies" / "spy_timing.py"
    exp.write_text(
        f'study = "{spy}:study"\nreason = "test"\n[search]\nsampler = "grid"\nbudget = 0\n'
        "[search.params]\nalpha = [0.001, 0.1, 10.0]\n"
    )
    root = tmp_path / "r"
    dry = invoke("search", str(exp), "--root", str(root), "--dry-run").output
    assert "3 candidate(s)" in dry
    out = json.loads(invoke("search", str(exp), "--root", str(root), "--json").stdout)
    assert len(out["candidates"]) == 3 and out["deflated"]["n_trials"] == 3
    assert "selected_oos_sharpe" in out["walk_forward_selection"] and out["pbo"]["pbo"] is not None
    assert len(Ledger(root / "ledger.jsonl").trials()) == 3


def _json_tail(text: str) -> dict:
    return json.loads(text[text.index("{") :])
