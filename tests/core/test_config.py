"""Site configuration: precedence, provenance, profiles, paths, and errors that name the key and the layer."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chairlift.core.config import ConfigError, load_config


def _w(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def machine(tmp_path: Path):
    """An isolated machine: system and user files under tmp, no project file, a fixed hostname."""

    def load(**kw):
        kw.setdefault("cwd", tmp_path / "work")
        kw.setdefault("env", {"HOME": str(tmp_path / "home")})
        kw.setdefault("system_file", tmp_path / "etc" / "config.toml")
        kw.setdefault("user_file", tmp_path / "home" / ".config" / "chairlift" / "config.toml")
        kw.setdefault("hostname", "box1")
        (tmp_path / "work").mkdir(exist_ok=True)
        return load_config(**kw)

    return load


def test_zero_configuration_resolves_to_xdg_defaults(machine):
    r = machine()
    paths = r.config.paths
    assert paths.home.name == "chairlift" and paths.store_dir() == paths.home / "store"
    assert paths.study_dir("slalom") == paths.home / "studies" / "slalom"
    assert r.profile is None and r.files == ()
    assert all(layer == "default" for _, _, layer in r.explain())


def test_precedence_system_user_project_profile_env_cli(machine, tmp_path: Path):
    _w(tmp_path / "etc" / "config.toml", '[paths]\nhome = "/sys/home"\nstore = "/sys/store"\n[compute]\nworkers = 2\n')
    _w(tmp_path / "home/.config/chairlift/config.toml", '[paths]\nhome = "/user/home"\n[compute]\nworkers = 4\n')
    _w(tmp_path / "work" / "chairlift.toml", "[compute]\nworkers = 6\n[profile.big]\ncompute.workers = 32\n")
    r = machine(
        env={"CHAIRLIFT_PATHS__REPORTS": "/env/reports"}, profile="big", overrides={"paths.cache": "/cli/cache"}
    )
    c = r.config
    assert str(c.paths.store_dir()) == "/sys/store"  # only the system file set it
    assert str(c.paths.home) == "/user/home"  # user beats system
    assert str(c.paths.study_dir("x")) == "/user/home/studies/x"  # derived from home
    assert c.compute.workers == 32  # profile beats project beats user
    assert str(c.paths.reports_dir()) == "/env/reports" and str(c.paths.cache_dir()) == "/cli/cache"
    prov = r.provenance
    assert prov["paths.store"].startswith("system") and prov["paths.home"].startswith("user")
    assert prov["compute.workers"] == "profile big" and prov["paths.reports"] == "env" and prov["paths.cache"] == "cli"


def test_env_beats_profile_and_values_are_typed(machine, tmp_path: Path):
    _w(tmp_path / "work" / "chairlift.toml", "[profile.p]\ncompute.workers = 8\n")
    r = machine(profile="p", env={"CHAIRLIFT_COMPUTE__WORKERS": "16", "CHAIRLIFT_SYSTEMD__ON_FAILURE": "false"})
    assert r.config.compute.workers == 16 and r.config.systemd.on_failure is False


def test_profile_selection_order(machine, tmp_path: Path):
    _w(
        tmp_path / "home/.config/chairlift/config.toml",
        '[profile.a]\ncompute.workers = 1\n[profile.b]\ncompute.workers = 2\n[profile.hosts]\n"box1" = "a"\n',
    )
    assert machine().profile == "a"  # host map
    assert machine(env={"CHAIRLIFT_PROFILE": "b"}).profile == "b"  # env beats host map
    assert machine(env={"CHAIRLIFT_PROFILE": "b"}, profile="a").profile == "a"  # explicit beats env
    with pytest.raises(ConfigError, match="'nope' is not defined"):
        machine(profile="nope")


def test_unknown_keys_and_wrong_types_name_the_key_and_layer(machine, tmp_path: Path):
    _w(tmp_path / "work" / "chairlift.toml", "[compute]\nworkres = 4\n")
    with pytest.raises(ConfigError, match=r"compute\.workres: unknown setting \(set by project"):
        machine()
    _w(tmp_path / "work" / "chairlift.toml", '[compute]\nworkers = "eight"\n')
    with pytest.raises(ConfigError, match=r"workers"):
        machine()
    _w(tmp_path / "work" / "chairlift.toml", "[colour]\nx = 1\n")
    with pytest.raises(ConfigError, match="colour: unknown setting"):
        machine()


def test_paths_expand_vars_and_resolve_relative_to_their_file(machine, tmp_path: Path):
    _w(
        tmp_path / "work" / "chairlift.toml",
        '[paths]\nhome = "rel/home"\nstore = "${SCRATCH}/store"\n[data]\nv = "~/vendor"\n',
    )
    r = machine(env={"SCRATCH": "/scratch", "HOME": "/home/me"})
    assert r.config.paths.home == (tmp_path / "work" / "rel" / "home").resolve()
    assert str(r.config.paths.store_dir()) == "/scratch/store"
    assert r.config.data["v"] == Path(os.path.expanduser("~/vendor"))


def test_undefined_variable_is_an_error(machine, tmp_path: Path):
    _w(tmp_path / "work" / "chairlift.toml", '[paths]\nhome = "${NOPE}/home"\n')
    with pytest.raises(ConfigError, match=r"paths\.home: \$\{NOPE\} is not defined"):
        machine()


def test_project_file_is_found_in_a_parent_directory(machine, tmp_path: Path):
    _w(tmp_path / "work" / "chairlift.toml", "[compute]\nworkers = 3\n")
    deep = tmp_path / "work" / "a" / "b"
    deep.mkdir(parents=True)
    assert machine(cwd=deep).config.compute.workers == 3


def test_record_is_plain_json_with_provenance(machine, tmp_path: Path):
    import json

    _w(tmp_path / "work" / "chairlift.toml", '[data]\nvendor = "/data/vendor"\n')
    rec = machine().record()
    json.dumps(rec)
    assert rec["values"]["data"]["vendor"] == "/data/vendor"
    assert rec["provenance"]["data.vendor"].startswith("project")


def test_explicit_config_file_from_env_is_a_layer(machine, tmp_path: Path):
    f = _w(tmp_path / "elsewhere.toml", "[compute]\nworkers = 9\n")
    r = machine(env={"CHAIRLIFT_CONFIG": str(f)})
    assert r.config.compute.workers == 9 and r.provenance["compute.workers"].startswith("CHAIRLIFT_CONFIG")


def test_explicit_config_file_that_does_not_exist_is_an_error(machine, tmp_path: Path):
    with pytest.raises(ConfigError, match="does not exist"):
        machine(env={"CHAIRLIFT_CONFIG": str(tmp_path / "missing.toml")})


def test_invalid_toml_names_the_layer_and_file(machine, tmp_path: Path):
    _w(tmp_path / "work" / "chairlift.toml", "[compute\nworkers = 1\n")
    with pytest.raises(ConfigError, match=r"project \(.*chairlift\.toml\): not valid TOML"):
        machine()


def test_malformed_env_names_are_errors(machine):
    with pytest.raises(ConfigError, match="CHAIRLIFT_<SECTION>__<KEY>"):
        machine(env={"CHAIRLIFT_WORKERS": "3"})


def test_data_alias_must_be_a_path_string(machine, tmp_path: Path):
    _w(tmp_path / "work" / "chairlift.toml", "[data]\nvendor = 3\n")
    with pytest.raises(ConfigError, match=r"data\.vendor: expected a path string"):
        machine()


def test_bare_dollar_variables_expand_too(machine, tmp_path: Path):
    _w(tmp_path / "work" / "chairlift.toml", '[paths]\nhome = "$SCRATCH/ch"\n')
    assert str(machine(env={"SCRATCH": "/s"}).config.paths.home) == "/s/ch"


def test_profile_tables_accept_nested_and_dotted_keys(machine, tmp_path: Path):
    _w(tmp_path / "work" / "chairlift.toml", '[profile.p]\ncompute.workers = 5\n[profile.p.paths]\nhome = "/p/home"\n')
    r = machine(profile="p")
    assert r.config.compute.workers == 5 and str(r.config.paths.home) == "/p/home"


def test_profile_must_be_a_table(machine, tmp_path: Path):
    _w(tmp_path / "work" / "chairlift.toml", 'profile = "laptop"\n')
    with pytest.raises(ConfigError, match="expected a table of profiles"):
        machine()


def test_secret_variables_are_not_config_settings(machine):
    r = machine(env={"CHAIRLIFT_SECRET_TOKEN": "x"})
    assert "token" not in str(r.record())


def test_starter_file_is_valid_and_changes_nothing(machine, tmp_path: Path):
    from chairlift.core.config import STARTER

    _w(tmp_path / "work" / "chairlift.toml", STARTER)
    assert all(layer == "default" for _, _, layer in machine().explain())
