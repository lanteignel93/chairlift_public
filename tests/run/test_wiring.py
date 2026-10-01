"""The CLI places studies under the configured home and shares one store: machine-independent, cross-study reuse."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from click.testing import CliRunner

from chairlift.run.cli import main

TOY = "chairlift.verify.toy:pipeline"


def machine_env(tmp_path: Path, name: str, **extra: str) -> dict[str, str]:
    """An isolated machine: its own XDG dirs and a user config pointing home somewhere of its own."""
    base = tmp_path / name
    cfg = base / "xdg_config" / "chairlift" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(f'[paths]\nhome = "{base / "chairlift_home"}"\n')
    env = {
        "XDG_CONFIG_HOME": str(base / "xdg_config"),
        "XDG_DATA_HOME": str(base / "xdg_data"),
        "XDG_CACHE_HOME": str(base / "xdg_cache"),
        "XDG_CONFIG_DIRS": str(base / "xdg_sys"),
        "HOME": str(base),
    }
    return env | extra


def cli(env: dict[str, str], *args: str, cwd: Path):
    with contextlib.chdir(cwd):
        return CliRunner(env=env).invoke(main, list(args), catch_exceptions=False)


def test_study_runs_under_home_studies_name_and_shares_the_store(tmp_path: Path):
    env = machine_env(tmp_path, "m1")
    out = cli(env, "run", TOY, "--json", cwd=tmp_path)
    assert out.exit_code == 0
    home = tmp_path / "m1" / "chairlift_home"
    assert list((home / "studies" / "toy" / "runs").glob("*.json"))
    assert any((home / "store").glob("manifest.*.jsonl")) and not (home / "studies" / "toy" / "store").exists()


def test_two_studies_reuse_each_others_stages_through_the_shared_store(tmp_path: Path):
    env = machine_env(tmp_path, "m1")
    cli(env, "run", TOY, cwd=tmp_path)
    second = json.loads(cli(env, "run", TOY, "--name", "toy-copy", "--json", cwd=tmp_path).output)
    assert {r["status"] for r in second} == {"hit"}  # a different study, same stages: nothing recomputed
    listed = cli(env, "runs", cwd=tmp_path)
    names = {line.split()[1] for line in listed.output.splitlines()[1:] if line.strip()}
    assert names == {"toy", "toy-copy"}


def test_same_experiment_on_two_machines_has_one_signature(tmp_path: Path):
    a = json.loads(cli(machine_env(tmp_path, "a"), "signature", TOY, "--json", cwd=tmp_path).output)
    b = json.loads(cli(machine_env(tmp_path, "b"), "signature", TOY, "--json", cwd=tmp_path).output)
    assert a["signature"] == b["signature"]


def test_run_record_carries_the_config_with_provenance_and_no_secret(tmp_path: Path):
    env = machine_env(tmp_path, "m1", CHAIRLIFT_SECRET_WEBHOOK="https://hooks.example/TOPSECRET")
    cli(env, "run", TOY, "--reason", "wiring", cwd=tmp_path)
    rec_path = next((tmp_path / "m1" / "chairlift_home" / "studies" / "toy" / "runs").glob("*.json"))
    text = rec_path.read_text()
    assert "TOPSECRET" not in text
    rec = json.loads(text)
    assert rec["meta"]["name"] == "toy"
    assert rec["meta"]["config"]["provenance"]["paths.home"].startswith("user")


def test_config_show_explain_and_check(tmp_path: Path):
    env = machine_env(tmp_path, "m1", CHAIRLIFT_COMPUTE__WORKERS="12")
    shown = cli(env, "config", "show", "--explain", cwd=tmp_path).output
    assert "compute.workers" in shown and "env" in shown and "user" in shown
    assert cli(env, "config", "check", cwd=tmp_path).exit_code == 0
    bad = machine_env(tmp_path, "m2", CHAIRLIFT_COMPUTE__WORKRES="1")
    result = CliRunner(env=bad).invoke(main, ["config", "check"])
    assert result.exit_code == 2 and "unknown setting" in result.output


def test_config_init_writes_a_starter_and_refuses_to_overwrite(tmp_path: Path):
    env = machine_env(tmp_path, "m1")
    runner = CliRunner(env=env)
    proj = tmp_path / "proj"
    proj.mkdir()
    with contextlib.chdir(proj):
        assert runner.invoke(main, ["config", "init", "--project"]).exit_code == 0
        assert Path("chairlift.toml").read_text().startswith("# chairlift site configuration")
        assert runner.invoke(main, ["config", "init", "--project"]).exit_code == 2
        assert runner.invoke(main, ["config", "check"]).exit_code == 0  # the starter itself is valid


def test_rerun_and_compare_find_runs_under_the_configured_home(tmp_path: Path):
    env = machine_env(tmp_path, "m1")
    cli(env, "run", TOY, "--set", "window=2", cwd=tmp_path)
    cli(env, "run", TOY, "--set", "window=4", cwd=tmp_path)
    runs = sorted((tmp_path / "m1" / "chairlift_home" / "studies" / "toy" / "runs").glob("*Z-*.json"))
    a, b = (json.loads(p.read_text())["run_id"] for p in runs)
    out = cli(env, "rerun", a, "--json", cwd=tmp_path)
    assert out.exit_code == 0 and json.loads(out.output)["ok"]
    assert len(list((tmp_path / "m1" / "chairlift_home" / "studies" / "toy" / "runs").glob("*Z-*.json"))) == 2
    rows = json.loads(cli(env, "compare", a, b, "--json", cwd=tmp_path).output)["rows"]
    assert {(r["key"], r["a"], r["b"]) for r in rows if r["section"] == "params"} == {("window", 2, 4)}
