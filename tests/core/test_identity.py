"""core/identity.py: code identity recorded on every build."""

from __future__ import annotations

import subprocess

import pytest

from chairlift.core import identity


def test_identity_names_the_package_and_a_commit():
    identity.code_identity.cache_clear()
    ident = identity.code_identity()
    assert ident.startswith("chairlift ")
    assert " @ " in ident or ident.endswith("(no git)")


def test_without_git_it_says_so_instead_of_failing(monkeypatch: pytest.MonkeyPatch):
    def boom(*_a: object, **_k: object) -> None:
        raise OSError("git not installed")

    identity.code_identity.cache_clear()
    monkeypatch.setattr(subprocess, "run", boom)
    try:
        assert identity.code_identity().endswith("(no git)")
    finally:
        identity.code_identity.cache_clear()


def test_dirty_tree_is_flagged(monkeypatch: pytest.MonkeyPatch):
    outputs = iter(["abcdef123456\n", " M src/x.py\n"])

    def fake(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout=next(outputs), stderr="")

    identity.code_identity.cache_clear()
    monkeypatch.setattr(subprocess, "run", fake)
    try:
        assert identity.code_identity().endswith("@ abcdef123456+dirty")
    finally:
        identity.code_identity.cache_clear()
