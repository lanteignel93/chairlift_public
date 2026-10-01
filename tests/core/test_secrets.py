"""core/secrets.py: credentials kept out of the config object, files must be private, values never serialize."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import pytest

from chairlift.core.config import load_config
from chairlift.core.secrets import SecretsError, load_secrets


def _w(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_secrets_file_must_be_private(tmp_path: Path):
    f = _w(tmp_path / "secrets.toml", 'webhook = "https://hooks.example/abc"\n')
    os.chmod(f, 0o644)
    with pytest.raises(SecretsError, match="chmod 600"):
        load_secrets(env={}, path=f)
    os.chmod(f, 0o600)
    s = load_secrets(env={"CHAIRLIFT_SECRET_SMTP": "pw"}, path=f)
    assert s["webhook"] == "https://hooks.example/abc" and s["smtp"] == "pw"
    assert "abc" not in repr(s) and "pw" not in repr(s)
    assert s.sources == {"webhook": str(f), "smtp": "env"}


def test_secrets_cannot_be_serialized_and_never_enter_the_config(tmp_path: Path):
    s = load_secrets(env={"CHAIRLIFT_SECRET_TOKEN": "s3cr3t"}, path=tmp_path / "none.toml")
    with pytest.raises(TypeError):
        pickle.dumps(s)
    with pytest.raises(TypeError):
        json.dumps(s)
    (tmp_path / "work").mkdir()
    r = load_config(
        cwd=tmp_path / "work",
        env={"CHAIRLIFT_SECRET_TOKEN": "s3cr3t", "HOME": str(tmp_path)},
        system_file=tmp_path / "none1.toml",
        user_file=tmp_path / "none2.toml",
        hostname="h",
    )  # secret env vars are not config settings
    assert "s3cr3t" not in json.dumps(r.record())


def test_missing_secrets_file_is_fine(tmp_path: Path):
    assert len(load_secrets(env={}, path=tmp_path / "absent.toml")) == 0


def test_env_overrides_the_file(tmp_path: Path):
    f = _w(tmp_path / "secrets.toml", 'token = "from-file"\n')
    os.chmod(f, 0o600)
    s = load_secrets(env={"CHAIRLIFT_SECRET_TOKEN": "from-env"}, path=f)
    assert s["token"] == "from-env" and s.sources["token"] == "env"


def test_non_string_secret_is_rejected(tmp_path: Path):
    f = _w(tmp_path / "secrets.toml", "port = 25\n")
    os.chmod(f, 0o600)
    with pytest.raises(SecretsError, match="must be a string"):
        load_secrets(env={}, path=f)
