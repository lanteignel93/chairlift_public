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
