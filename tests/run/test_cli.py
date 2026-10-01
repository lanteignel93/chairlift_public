import json

from click.testing import CliRunner

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
